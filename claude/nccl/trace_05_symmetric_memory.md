# NCCL Symmetric Memory Execution Trace

Complete literate code walkthrough of NCCL's symmetric memory feature using Example 05.

**Example Source**: [/home/jeromeku/torchcomms/thirdparty/nccl/examples/05_symmetric_memory/01_allreduce/main.cc](../../../thirdparty/nccl/examples/05_symmetric_memory/01_allreduce/main.cc)

---

## Table of Contents

1. [Overview](#overview)
2. [What is Symmetric Memory?](#what-is-symmetric-memory)
3. [Complete Execution Flow](#complete-execution-flow)
4. [Detailed API Traces](#detailed-api-traces)
5. [Symmetric Memory Deep Dive](#symmetric-memory-deep-dive)
6. [Performance Optimizations](#performance-optimizations)
7. [Data Structures](#data-structures)
8. [Sequence Diagrams](#sequence-diagrams)

---

## Overview

### What Symmetric Memory Is

**Symmetric memory** in NCCL refers to memory buffers that are registered across all ranks with a special flag (`NCCL_WIN_COLL_SYMMETRIC`) that enables NCCL to:

1. **Create a unified virtual address space** across all GPUs using CUDA's Virtual Memory Management (VMM) APIs
2. **Enable direct peer-to-peer memory access** with predictable address layouts
3. **Use specialized high-performance kernels** that exploit symmetric memory properties
4. **Leverage multicast memory (NVLS)** for even faster collective operations on supported hardware

### Differences from Regular Buffer Registration (Example 04)

| Feature | Regular Registration (`ncclCommRegister`) | Symmetric Memory (`ncclCommWindowRegister` + `NCCL_WIN_COLL_SYMMETRIC`) |
|---------|-------------------------------------------|-------------------------------------------------------------------------|
| **API** | `ncclCommRegister(comm, buff, size, &handle)` | `ncclCommWindowRegister(comm, buff, size, &win, NCCL_WIN_COLL_SYMMETRIC)` |
| **Return Type** | Opaque `void* handle` | `ncclWindow_t` (pointer to `ncclWindow_vidmem`) |
| **Memory Layout** | Each rank's buffer at independent addresses | Coordinated "flat" virtual address space across ranks |
| **Peer Access** | Through normal P2P mechanisms | Direct access via symmetric VA mapping |
| **Kernel Selection** | Standard NCCL kernels | Specialized symmetric kernels in `src/device/symmetric/` |
| **Multicast Support** | No | Yes (NVLS on Hopper+) |
| **Rank Coordination** | Local registration only | Bootstrap exchange of memory handles across ranks |

### Performance Benefits

1. **Lower latency**: Direct memory access without indirection
2. **Higher bandwidth**: Multicast operations on NVLS-capable hardware
3. **Better scaling**: Optimized kernels that leverage symmetric layout
4. **Reduced overhead**: Single kernel launch handles all peer communication

---

## Complete Execution Flow

### User Code Entry Point

From [main.cc:101-104](../../../thirdparty/nccl/examples/05_symmetric_memory/01_allreduce/main.cc#L101):

```cpp
// Register symmetric memory windows with NCCL
ncclWindow_t send_win;
ncclWindow_t recv_win;
NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win,
                                 NCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win,
                                 NCCL_WIN_COLL_SYMMETRIC));
```

**Key Observations**:
- Uses `ncclCommWindowRegister` instead of `ncclCommRegister`
- Passes `NCCL_WIN_COLL_SYMMETRIC` flag to enable symmetric memory optimizations
- Returns `ncclWindow_t` handles (typedef for `ncclWindow_vidmem*`)

### AllReduce with Symmetric Memory

From [main.cc:134-135](../../../thirdparty/nccl/examples/05_symmetric_memory/01_allreduce/main.cc#L134):

```cpp
// Perform AllReduce operation
// Since symmetric memory is registered, NCCL can apply optimized algorithms
NCCLCHECK(ncclAllReduce(d_sendbuff, d_recvbuff, count, ncclFloat, ncclSum,
                        comm, stream));
```

**What happens**: NCCL detects that buffers are registered symmetric windows and selects optimized kernels.

### Cleanup

From [main.cc:193-194](../../../thirdparty/nccl/examples/05_symmetric_memory/01_allreduce/main.cc#L193):

```cpp
// Deregister symmetric memory windows from communicator
NCCLCHECK(ncclCommWindowDeregister(comm, send_win));
NCCLCHECK(ncclCommWindowDeregister(comm, recv_win));
```

---

## Detailed API Traces

### 1. ncclCommWindowRegister with NCCL_WIN_COLL_SYMMETRIC

#### Entry Point: Public API

**File**: [src/dev_runtime.cc:891-924](../../../thirdparty/nccl/src/dev_runtime.cc#L891)

```cpp
NCCL_API(ncclResult_t, ncclCommWindowRegister, ncclComm_t comm, void* ptr, size_t size, ncclWindow_t* win, int winFlags);
ncclResult_t ncclCommWindowRegister(
    struct ncclComm* comm, void* userPtr, size_t userSize,
    struct ncclWindow_vidmem** outWinDev, int winFlags
  ) {
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  struct ncclDevrRegTask* task;

  CUDACHECK(cudaGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());  // Enter group context

  if (userPtr == nullptr || userSize == 0 || !(comm->symmetricSupport || ncclParamLocalRegister()))
    goto exit;

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);  // Initialize symmetric memory subsystem

  // Create task for deferred execution
  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  task->userPtr = userPtr;
  task->userSize = userSize;
  task->winFlags = winFlags;
  task->outWinDev = outWinDev;
  ncclIntruQueueEnqueue(&comm->devrState.regTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);  // Join group collective
```

**Key Steps**:
1. Enters NCCL group context (for collective synchronization)
2. Validates symmetric support
3. Initializes symmetric memory state if needed
4. Enqueues registration task for group execution
5. Actual registration happens in `ncclGroupEnd()`

#### Symmetric Memory Initialization

**File**: [src/dev_runtime.cc:57-106](../../../thirdparty/nccl/src/dev_runtime.cc#L57)

```cpp
ncclResult_t ncclDevrInitOnce(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  if (devr->bigSize != 0) return ncclSuccess;  // Already initialized

  // LSA (Local Symmetric Access) team: consecutive ranks that can share symmetric VA
  int lsaSize = 0;
  int nodeSize = 1;
  for (int r=1; r < comm->nRanks; r++) {
    if (comm->rankToNode[r] == comm->rankToNode[r-1]) {
      nodeSize += 1;
    } else {
      lsaSize = gcd(lsaSize, nodeSize);
      nodeSize = 1;
    }
  }
  lsaSize = gcd(lsaSize, nodeSize);
  devr->lsaSize = lsaSize;  // Number of ranks in LSA team
  devr->lsaSelf = comm->rank % lsaSize;
  devr->lsaRankList = (int*)malloc(devr->lsaSize*sizeof(int));
  for (int i=0; i < devr->lsaSize; i++) {
    devr->lsaRankList[i] = comm->rank + (i - devr->lsaSelf);
  }

  // Get CUDA VMM allocation granularity
  CUmemAllocationProp memProp = {};
  memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  memProp.requestedHandleTypes = ncclCuMemHandleType;
  memProp.location.id = comm->cudaDev;
  CUCHECKGOTO(cuMemGetAllocationGranularity(&devr->granularity, &memProp,
              CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), ret, fail_lsaRankList);

  // Calculate "big VA space" size - large enough for any GPU's memory
  devr->bigSize = ncclParamWinStride();
  if (-devr->bigSize <= 1) {
    devr->bigSize = 1;
    for (int r=0; r < comm->nRanks; ++r) {
      devr->bigSize = std::max<size_t>(devr->bigSize, comm->peerInfo[r].totalGlobalMem);
    }
  }
  devr->bigSize = alignUp(devr->bigSize, size_t(1)<<32);  // Align to 4GB
  INFO(NCCL_INIT, "Symmetric VA size=%ldGB", (long)devr->bigSize>>30);

  ncclSpaceConstruct(&devr->bigSpace);  // Allocator for offsets in big VA
  ncclShadowPoolConstruct(&devr->shadows);  // Host/device shadow memory pool
  return ncclSuccess;
```

**Critical Concept - "Big VA Space"**:
- NCCL creates a virtual address space large enough to hold any GPU's entire memory
- Default: max GPU memory size across all ranks, aligned to 4GB
- Each rank reserves `lsaSize * bigSize` bytes of VA space
- Within this space, rank i's memory is at offset `i * bigSize`
- This creates a **symmetric flat address space** across the LSA team

#### Window Registration During Group Execution

**File**: [src/dev_runtime.cc:578-648](../../../thirdparty/nccl/src/dev_runtime.cc#L578)

```cpp
ncclResult_t ncclDevrWindowRegisterInGroup(
    struct ncclComm* comm,
    void* userPtr, size_t userSize, int winFlags, ncclWindow_t* outWinDev
  ) {
  ncclResult_t ret = ncclSuccess;
  CUdeviceptr memAddr = 0;
  size_t memSize = 0;
  CUmemGenericAllocationHandle memHandle = 0x0;
  size_t memOffset;
  struct ncclDevrMemory* mem = nullptr;
  cudaStream_t stream = nullptr;
  void* localRegHandle = nullptr;

  // Step 1: Local buffer registration (for IPC, RDMA, etc.)
  NCCLCHECKGOTO(ncclCommRegister(comm, userPtr, userSize, &localRegHandle), ret, fail);

  if (!comm->symmetricSupport) {
    // Fallback: just return local registration handle
    *outWinDev = reinterpret_cast<struct ncclWindow_vidmem*>(localRegHandle);
    return ncclSuccess;
  }

  // Step 2: Initialize symmetric kernel support if NCCL_WIN_COLL_SYMMETRIC flag set
  if (winFlags & NCCL_WIN_COLL_SYMMETRIC) {
    NCCLCHECKGOTO(ncclSymkInitOnce(comm), ret, fail);
  }

  // Step 3: Get underlying CUDA memory handle
  CUCHECKGOTO(cuMemGetAddressRange(&memAddr, &memSize,
              reinterpret_cast<CUdeviceptr>(userPtr)), ret, fail_locReg);
  memOffset = reinterpret_cast<CUdeviceptr>(userPtr) - memAddr;
  if (memOffset%NCCL_WIN_REQUIRED_ALIGNMENT != 0) {
    WARN("Window address must be suitably aligned.");
    ret = ncclInvalidArgument;
    goto fail;
  }

  CUCHECKGOTO(cuMemRetainAllocationHandle(&memHandle, reinterpret_cast<void*>(memAddr)),
              ret, fail_locReg);

  // Step 4: Register memory with symmetric subsystem (collective operation!)
  NCCLCHECKGOTO(symMemoryObtain(comm, memHandle, (void*)memAddr, memSize, &mem),
                ret, fail_locReg_memHandle);
  memHandle = 0x0; // symMemoryObtain took our reference

  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail);

  // Step 5: Create window structure
  NCCLCHECKGOTO(symWindowCreate(
      comm, mem, memOffset, userPtr, userSize, winFlags, localRegHandle, outWinDev, nullptr, stream
    ), ret, fail_locReg_memHandle_mem_stream);
  mem = nullptr; // symWindowCreate took our reference

  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail_locReg_memHandle_mem_stream_win);

  // Step 6: Barrier - all ranks must complete registration
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef),
                ret, fail_locReg_memHandle_mem_stream_win);

  cudaStreamDestroy(stream);
  return ret;
```

**Key Points**:
1. **Local registration first**: Standard `ncclCommRegister` for IPC/RDMA
2. **CUDA VMM handle extraction**: Gets the underlying `CUmemGenericAllocationHandle`
3. **Collective memory registration**: `symMemoryObtain` exchanges handles across ranks
4. **Window creation**: Creates device-side window structure
5. **Barrier synchronization**: Ensures all ranks complete before returning

#### Memory Handle Exchange and Mapping

**File**: [src/dev_runtime.cc:360-424](../../../thirdparty/nccl/src/dev_runtime.cc#L360)

```cpp
static ncclResult_t symMemoryObtain(
    struct ncclComm* comm, CUmemGenericAllocationHandle memHandle, void* memAddr, size_t size,
    struct ncclDevrMemory** outMem
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  int64_t bigOffset = 0;

  struct ncclDevrMemory* mem = devr->memHead;
  while (mem != nullptr) {
    if (mem->memHandle == memHandle) {
      CUCHECKIGNORE(cuMemRelease(memHandle));  // Already registered
      goto leave;
    }
    mem = mem->next;
  }

  // New memory - create tracking structure
  mem = (struct ncclDevrMemory*)malloc(sizeof(struct ncclDevrMemory));
  mem->refCount = 0;
  mem->memHandle = memHandle;
  mem->primaryAddr = memAddr;
  mem->size = size;

  // Allocate offset in the big VA space
  NCCLCHECKGOTO(ncclSpaceAlloc(&devr->bigSpace, devr->bigSize, size, devr->granularity, &bigOffset),
                ret, fail_mem);
  mem->bigOffset = bigOffset;

  // Map memory into flat VA space for LSA team (COLLECTIVE OPERATION)
  NCCLCHECKGOTO(symMemoryMapLsaTeam(comm, memHandle, size, bigOffset), ret, fail_mem_space);

  // If caller doesn't have a VA, use the LSA mapping
  if (mem->primaryAddr == nullptr) {
    mem->primaryAddr = (char*)devr->lsaFlatBase + devr->lsaSelf*devr->bigSize + mem->bigOffset;
  }

  // Bind to existing multicast teams if any
  for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
    NCCLCHECKGOTO(symBindTeamMemory(comm, t, mem), ret, fail_mem_space_teams);
  }

  // Register with GIN (GPU Initiated Network) if enabled
  if (devr->ginEnabled) {
    NCCLCHECKGOTO(symMemoryRegisterGin(comm, mem), ret, fail_mem_space_teams);
  }

  // Add to list of registered memories
  mem->next = devr->memHead;
  devr->memHead = mem;

leave:
  mem->refCount += 1;
  *outMem = mem;
  return ret;
```

**Critical Function**: `symMemoryMapLsaTeam` performs the collective memory mapping.

**File**: [src/dev_runtime.cc:147-203](../../../thirdparty/nccl/src/dev_runtime.cc#L147)

```cpp
static ncclResult_t symMemoryMapLsaTeam(
    struct ncclComm* comm, CUmemGenericAllocationHandle memHandle, size_t size, size_t bigOffset
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  CUmemAccessDesc accessDesc = {};
  union Message {
    CUmemGenericAllocationHandle memHandle;
    CUmemFabricHandle fabricHandle;
  };

  Message* messages = (Message*)calloc(devr->lsaSize, sizeof(Message));

  // Export my memory handle for sharing
  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    messages[devr->lsaSelf].memHandle = memHandle;
  } else {
    CUCHECKGOTO(cuMemExportToShareableHandle(&messages[devr->lsaSelf].fabricHandle,
                memHandle, ncclCuMemHandleType, 0), ret, fail);
  }

  // COLLECTIVE: AllGather memory handles across LSA team
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, devr->lsaRankList, devr->lsaSelf,
                devr->lsaSize, messages, sizeof(Message)), ret, fail);

  // Reserve flat VA space on first use
  if (devr->lsaFlatBase == nullptr) {
    CUdeviceptr addr;
    CUCHECKGOTO(cuMemAddressReserve(&addr, devr->lsaSize*devr->bigSize, NCCL_MAX_PAGE_SIZE, 0, 0),
                ret, fail);
    devr->lsaFlatBase = reinterpret_cast<void*>(addr);
  }

  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = comm->cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  // Map each rank's memory at the correct offset in flat VA space
  for (int r = 0; r < devr->lsaSize; r++) {
    CUmemGenericAllocationHandle impHandle;
    if (r == devr->lsaSelf) {
      impHandle = memHandle;  // My own memory
    } else {
      // Import peer's memory handle
      if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        int fd = -1;
        NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, devr->lsaRankList[r], &messages[r], &fd),
                      ret, fail);
        CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle,
                    reinterpret_cast<void*>((uintptr_t)fd), ncclCuMemHandleType), ret, fail);
        SYSCHECKGOTO(close(fd), "close", ret, fail);
      } else {
        CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle, (void*)&messages[r].fabricHandle,
                    ncclCuMemHandleType), ret, fail);
      }
    }

    // Map at offset: rank * bigSize + allocation offset
    CUdeviceptr addr = reinterpret_cast<uintptr_t>((char*)devr->lsaFlatBase + r*devr->bigSize + bigOffset);
    CUCHECKGOTO(cuMemMap(addr, size, 0, impHandle, 0), ret, fail);
    CUCHECKGOTO(cuMemSetAccess(addr, size, &accessDesc, 1), ret, fail);

    if (r != devr->lsaSelf) {
      CUCHECKGOTO(cuMemRelease(impHandle), ret, fail);  // Can release after mapping
    }
  }

  // Barrier: ensure everyone has imported my handle before I can release it
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, devr->lsaRankList, devr->lsaSelf,
                devr->lsaSize, 0xbeef), ret, fail);
```

**Memory Layout After Mapping**:

```
devr->lsaFlatBase:
┌─────────────────────────────────────────────────────────────┐
│ Rank 0's memory space (bigSize bytes)                       │
│  [buffer1 at bigOffset1] [buffer2 at bigOffset2] ...       │
├─────────────────────────────────────────────────────────────┤
│ Rank 1's memory space (bigSize bytes)                       │
│  [buffer1 at bigOffset1] [buffer2 at bigOffset2] ...       │
├─────────────────────────────────────────────────────────────┤
│ Rank 2's memory space (bigSize bytes)                       │
│  [buffer1 at bigOffset1] [buffer2 at bigOffset2] ...       │
└─────────────────────────────────────────────────────────────┘

Address calculation for rank r's buffer:
  devr->lsaFlatBase + r * bigSize + bigOffset
```

#### Window Structure Creation

**File**: [src/dev_runtime.cc:463-536](../../../thirdparty/nccl/src/dev_runtime.cc#L463)

```cpp
static ncclResult_t symWindowCreate(
    struct ncclComm* comm, struct ncclDevrMemory* mem,
    size_t memOffset, void* userPtr, size_t userSize, int winFlags, void* localReg,
    struct ncclWindow_vidmem** outWinDev, struct ncclDevrWindow** outWin,
    cudaStream_t stream
  ) {
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevrWindow* win;

  // Host-side window tracking structure
  win = (struct ncclDevrWindow*)malloc(sizeof(struct ncclDevrWindow));
  memset(win, 0, sizeof(*win));
  win->memory = mem;
  win->size = userSize;
  win->bigOffset = mem->bigOffset + memOffset;
  win->winFlags = winFlags;
  win->localRegHandle = localReg;
  if (userPtr == nullptr) {
    win->userPtr = (char*)devr->lsaFlatBase + (devr->lsaSelf*devr->bigSize) + mem->bigOffset;
  } else {
    win->userPtr = userPtr;
  }

  // Device-side window structure
  struct ncclWindow_vidmem* winDev;
  struct ncclWindow_vidmem* winDevHost;
  NCCLCHECK(ncclShadowPoolAlloc(&devr->shadows, &winDev, &winDevHost, stream));
  win->vidmem = winDev;

  // Populate device window fields
  winDevHost->lsaFlatBase = (char*)devr->lsaFlatBase + win->bigOffset;  // Points to rank 0's buffer
  winDevHost->mcOffset4K = win->bigOffset>>12;  // Multicast offset in 4KB units
  winDevHost->stride4G = devr->bigSize>>32;     // Stride between ranks in 4GB units
  winDevHost->lsaRank = devr->lsaSelf;
  winDevHost->worldRank = comm->rank;
  winDevHost->winHost = (void*)win;             // Backpointer to host structure
  winDevHost->ginOffset4K = memOffset>>12;      // GIN offset
  for (int i=0; i < NCCL_GIN_MAX_CONTEXTS; i++) {
    winDevHost->ginWins[i] = mem->ginDevWins[i];
  }

  // Copy to device
  CUDACHECK(cudaMemcpyAsync(winDev, winDevHost, sizeof(struct ncclWindow_vidmem),
            cudaMemcpyHostToDevice, stream));

  // Add to window lookup table (for finding windows from buffer pointers)
  NCCLCHECK(symWindowTableInitOnce(comm, stream));
  struct ncclDevCommWindowTable* tableDev = devr->windowTable;
  while (true) {
    struct ncclDevCommWindowTable* tableHost;
    NCCLCHECK(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost));
    int i = 0;
    while (i < 32 && tableHost->entries[i].window != nullptr) i += 1;
    if (i < 32) {
      tableHost->entries[i].base = userAddr;
      tableHost->entries[i].size = userSize;
      tableHost->entries[i].window = winDev;
      CUDACHECK(cudaMemcpyAsync(&tableDev->entries[i], &tableHost->entries[i],
                sizeof(tableHost->entries[i]), cudaMemcpyHostToDevice, stream));
      break;
    }
    if (tableHost->next == nullptr) {
      NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(&devr->shadows, &tableHost->next,
                nullptr, stream));
      CUDACHECK(cudaMemcpyAsync(&tableDev->next, &tableHost->next, sizeof(tableHost->next),
                cudaMemcpyHostToDevice, stream));
    }
    tableDev = tableHost->next;
  }

  // Insert into sorted window list for fast lookup
  int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted,
                            devr->winSortedCount, userAddr);
  struct ncclDevrWindowSorted winSort;
  winSort.userAddr = userAddr;
  winSort.size = userSize;
  winSort.win = win;
  listInsert(&devr->winSorted, &devr->winSortedCapacity, &devr->winSortedCount, i, winSort);

  if (outWinDev) *outWinDev = winDev;
  if (outWin) *outWin = win;
  return ncclSuccess;
}
```

---

### 2. AllReduce with Symmetric Memory

#### Detection and Kernel Selection

**File**: [src/enqueue.cc:2573-2581](../../../thirdparty/nccl/src/enqueue.cc#L2573)

```cpp
struct ncclDevrWindow* sendWin;
struct ncclDevrWindow* recvWin;
ncclDevrFindWindow(comm, info->sendbuff, &sendWin);
ncclDevrFindWindow(comm, info->recvbuff, &recvWin);

// Check if we can use symmetric kernels
if (comm->symmetricSupport && comm->nNodes == 1 && sendWin && recvWin &&
    (sendWin->winFlags & recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC) &&
    comm->config.CTAPolicy == NCCL_CTA_POLICY_ZERO && ceImplemented) {
  NCCLCHECK(ceCollTaskAppend(comm, info, sendWin, recvWin, opDev));
}
```

**Key Check**: Both buffers must be registered symmetric windows with `NCCL_WIN_COLL_SYMMETRIC` flag.

#### Symmetric Kernel Selection

**File**: [src/sym_kernels.cc:327-362](../../../thirdparty/nccl/src/sym_kernels.cc#L327)

```cpp
ncclResult_t ncclSymkPickKernel(
    struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty,
    size_t nEltsTotal, size_t nEltsMax, int nWorks,
    float* estTimeUs, ncclSymkKernelId* kernelId, int* nBlocks, int* nWarps
  ) {
  uint32_t kmask = ncclSymkMask(comm, coll, red, ty, nEltsMax);

  // We currently don't support grouping for LL kernels.
  if (nWorks > 1)
    kmask &= ~kernelMask_LL;

  ncclSymkKernelId bestKernel = ncclSymkKernelId_Count;
  float bestTime = 1.e30f;
  int bestBlocks = 999;
  size_t nBytes = nEltsTotal*ncclTypeSize(ty);

  constexpr float smPenalty = .025f;  // 2.5% penalty per SM used
  uint32_t kmaskRemain = kmask;

  // Evaluate each candidate kernel
  while (kmaskRemain != 0) {
    ncclSymkKernelId k = (ncclSymkKernelId)popFirstOneBit(&kmaskRemain);
    float kTime;
    int kBlocks;
    queryModel(comm, k, nBytes, &kTime, &kBlocks);  // Performance model

    if (kTime*(1.0f + smPenalty*kBlocks) < bestTime*(1.0f + smPenalty*bestBlocks)) {
      bestKernel = k;
      bestTime = kTime;
      bestBlocks = kBlocks;
    }
  }

  *kernelId = bestKernel;
  *estTimeUs = kmask==0 || kernelMask_user() == (1<<ncclSymkKernelId_Count)-1 ? bestTime : 0.0f;
  *nBlocks = bestBlocks;
  *nWarps = 16;
  return ncclSuccess;
}
```

**Available Kernels for AllReduce**:

From [src/include/sym_kernels.h:27-32](../../../thirdparty/nccl/src/include/sym_kernels.h#L27):

```cpp
enum ncclSymkKernelId {
  ncclSymkKernelId_AllReduce_AGxLL_R,          // AllGather + LL (Low Latency) + Reduce
  ncclSymkKernelId_AllReduce_AGxLLMC_R,        // AllGather + LL Multicast + Reduce
  ncclSymkKernelId_AllReduce_RSxLD_AGxST,      // ReduceScatter + LD + AllGather + ST
  ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC,  // With multicast (NVLS)
  ncclSymkKernelId_AllReduce_RSxNet_ARxMC_AGxNet,
  ...
};
```

**Kernel Selection Logic**:

From [src/sym_kernels.cc:275-317](../../../thirdparty/nccl/src/sym_kernels.cc#L275):

```cpp
static uint32_t ncclSymkMask(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red,
                              ncclDataType_t ty, size_t nElts) {
  uint32_t kmask = kernelMask_coll(coll);
  kmask &= kernelMask_user();  // User can restrict via NCCL_SYM_KERNEL env var

  bool hasSTMC = comm->nvlsSupport;  // Store multicast (NVLS)
  bool hasLDMC = false;              // Load multicast
  if (comm->nvlsSupport) {
    switch (ty) {
    case ncclInt32:
    case ncclUint32:
    case ncclInt64:
    case ncclUint64:
    case ncclFloat16:
    case ncclBfloat16:
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax;
      break;
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax;
      hasLDMC &= comm->compCap >= 100;  // Hopper+
      break;
    case ncclFloat:
    case ncclDouble:
      hasLDMC = red == ncclDevSum;
      break;
    default: break;
    }
  }
  if (!hasSTMC) kmask &= ~kernelMask_STMC;
  if (!hasLDMC) kmask &= ~kernelMask_LDMC;

  size_t nBytes = nElts*ncclTypeSize(ty);
  size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks)*nBytes;
  // LL kernels use 32-bit ints for element counts
  if (nBusBytes >= (size_t(2)<<30)) kmask &= ~kernelMask_LL;
  // Any kernel might use 32-bit int for loop chunks
  if (nBusBytes >= 32*(size_t(2)<<30)) kmask = 0;

  if (comm->nNodes > 1) kmask &= ~kernelMask_LSA;  // Multi-node: no LSA-only kernels

  return kmask;
}
```

#### Device Kernel Execution: AllReduce Deep Dive

**File**: [src/device/symmetric/all_reduce.cuh:7-120](../../../thirdparty/nccl/src/device/symmetric/all_reduce.cuh#L7)

```cuda
template<int BytePerPack, int UnrollPacks, int UnrollPeers, typename T, typename Red>
static __device__ __forceinline__ void allreduceDeep(
    ncclSymkArgsHandler const& handler, int tn, int t,
    bool waitNeeded, ncclLsaBarrierSession<ncclCoopCta>& bar,
    Red red, ncclSymPtr<char> input, ncclSymPtr<char> output, int32_t nIters
  ) {
  using Pack = BytePack<BytePerPack>;
  using Acc = typename Red::EltType;
  using AccPack = BytePack<BytePerPack*sizeof(Acc)/sizeof(T)>;

  ncclTeam world = ncclTeamWorld(handler.comm);
  int wn = tn/WARP_SIZE;
  int w = t/WARP_SIZE;
  int lane = t%WARP_SIZE;
  int const& rank = handler.comm.rank;
  int const& nRanks = handler.comm.nRanks;

  // Symmetric pointer allows direct access to any rank's memory
  ncclSymPtr<Pack> inpPacks = (ncclSymPtr<Pack>)input + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  ncclSymPtr<Pack> outPacks = (ncclSymPtr<Pack>)output + intptr_t(w)*UnrollPacks*WARP_SIZE + lane;
  Pack acc0[UnrollPacks];

  nIters -= w;
  if (0 < nIters) {
    // Load from my own rank's buffer using symmetric pointer
    #pragma unroll
    for (int u=0; u < UnrollPacks; u++) {
      acc0[u] = inpPacks.peerPtr(world, rank)[u*WARP_SIZE];  // ← DIRECT PEER ACCESS
    }
  }

  if (waitNeeded) bar.wait(ncclCoopCta(), cuda::memory_order_relaxed);

  if (0 < nIters) {
    while (true) {
      AccPack acc1[UnrollPacks];
      int r = rank;
      if (++r == nRanks) r = 0;

      { Pack tmp1[UnrollPacks];
        // Load from next rank's buffer
        #pragma unroll
        for (int u=0; u < UnrollPacks; u++) {
          tmp1[u] = inpPacks.peerPtr(world, r)[u*WARP_SIZE];  // ← DIRECT PEER ACCESS
        }
        #pragma unroll
        for (int u=0; u < UnrollPacks; u++) {
          acc1[u] = applyReduce(red, applyCast<T, Acc>(acc0[u]), applyCast<T, Acc>(tmp1[u]));
        }
      }

      if (++r == nRanks) r = 0;

      int dr = 2;
      // Unrolled loop to read from multiple peers
      #pragma unroll 2
      for (int partial=0; partial <= 1; partial++) {
        #pragma unroll 1
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          if (partial && dr == nRanks) break;

          Pack tmp1[UnrollPeers][UnrollPacks];
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && ur!=0 && dr+ur == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              tmp1[ur][u] = inpPacks.peerPtr(world, r)[u*WARP_SIZE];  // ← DIRECT PEER ACCESS
            }
            if (++r == nRanks) r = 0;
          }
          // Reduce fetched data
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && ur!=0 && dr+ur == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              acc1[u] = applyReduce(red, acc1[u], applyCast<T, Acc>(tmp1[ur][u]));
            }
          }
        }
      }

      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) acc0[u] = applyCast<Acc, T>(acc1[u]);

      // Write results to all ranks
      dr = 0;
      r = rank;
      #pragma unroll 2
      for (int partial=0; partial <= 1; partial++) {
        #pragma unroll 1
        for (int i = 0;
             partial ? i < 1 : (dr + UnrollPeers <= nRanks);
             partial ? i++ : (dr += UnrollPeers)) {
          #pragma unroll
          for (int ur=0; ur < UnrollPeers-partial; ur++) {
            if (partial && dr == nRanks) break;
            #pragma unroll UnrollPacks
            for (int u=0; u < UnrollPacks; u++) {
              outPacks.peerPtr(world, r)[u*WARP_SIZE] = acc0[u];  // ← DIRECT PEER WRITE
            }
            if (++r == nRanks) r = 0;
          }
        }
      }

      // Advance pointers for next iteration
      inpPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      outPacks += intptr_t(wn)*UnrollPacks*WARP_SIZE;
      nIters -= wn;
      if (nIters <= 0) break;

      // Load data for next iteration
      #pragma unroll
      for (int u=0; u < UnrollPacks; u++) {
        acc0[u] = inpPacks.peerPtr(world, rank)[u*WARP_SIZE];
      }
    }
  }
}
```

**Critical Feature: `ncclSymPtr::peerPtr()`**

This is the magic that makes symmetric memory work. Let's look at its implementation:

**File**: [src/include/nccl_device/impl/ptr__funcs.h](../../../thirdparty/nccl/src/include/nccl_device/impl/ptr__funcs.h) (inferred from usage)

```cuda
template<typename T>
struct ncclSymPtr {
  ncclWindow_vidmem* win;
  size_t offset;

  __device__ T* peerPtr(ncclTeam team, int peerRank) const {
    // Calculate address in peer's memory using symmetric layout
    // Base address for rank 0's buffer
    char* base = win->lsaFlatBase;

    // Stride between ranks (in 4GB units, converted to bytes)
    size_t stride = (size_t)win->stride4G << 32;

    // Address = base + peerRank * stride + offset
    return (T*)(base + peerRank * stride + offset);
  }

  __device__ T* mcPtr() const {
    // Multicast address calculation for NVLS
    // Using multicast offset stored in window
    return /* multicast base + mcOffset4K */;
  }
};
```

**Address Calculation Example**:

For a 4-rank job with `bigSize = 128GB`:
- Rank 0's buffer at `lsaFlatBase + 0*128GB + bigOffset`
- Rank 1's buffer at `lsaFlatBase + 1*128GB + bigOffset`
- Rank 2's buffer at `lsaFlatBase + 2*128GB + bigOffset`
- Rank 3's buffer at `lsaFlatBase + 3*128GB + bigOffset`

Each rank can directly read/write any other rank's buffer!

---

### 3. ncclCommWindowDeregister

**File**: [src/dev_runtime.cc:927-950](../../../thirdparty/nccl/src/dev_runtime.cc#L927)

```cpp
NCCL_API(ncclResult_t, ncclCommWindowDeregister, ncclComm_t comm, ncclWindow_t win);
ncclResult_t ncclCommWindowDeregister(struct ncclComm* comm, struct ncclWindow_vidmem* winDev) {
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  cudaStream_t stream;

  if (winDev == nullptr) goto exit;

  if (!comm->symmetricSupport) {
    // Fallback: just deregister local handle
    NCCLCHECKGOTO(ncclCommDeregister(comm, winDev), ret, fail);
    goto exit;
  }

  CUDACHECKGOTO(cudaGetDevice(&saveDev), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), ret, fail_dev);

  // Destroy window and cleanup mappings
  NCCLCHECKGOTO(symWindowDestroy(comm, winDev, stream), ret, fail_dev_stream);

fail_dev_stream:
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
fail_dev:
  cudaSetDevice(saveDev);
fail:
exit:
  return ret;
}
```

**File**: [src/dev_runtime.cc:538-576](../../../thirdparty/nccl/src/dev_runtime.cc#L538)

```cpp
static ncclResult_t symWindowDestroy(struct ncclComm* comm, struct ncclWindow_vidmem* winDev,
                                      cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclWindow_vidmem* winDevHost;
  struct ncclDevrWindow* winHost;

  // Get host-side window structure
  NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, winDev, &winDevHost), ret, fail);
  winHost = (struct ncclDevrWindow*)winDevHost->winHost;

  // Drop reference on memory (may trigger unmapping if last reference)
  symMemoryDropRef(comm, winHost->memory);

  // Remove from window lookup table
  { struct ncclDevCommWindowTable* tableDev = devr->windowTable;
    while (true) {
      struct ncclDevCommWindowTable* tableHost;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost), ret, remove_winSorted);
      int i = 0;
      while (i < 32 && tableHost->entries[i].window != winDev) i += 1;
      if (i < 32) {
        memset(&tableHost->entries[i], 0, sizeof(tableHost->entries[i]));
        CUDACHECKGOTO(cudaMemsetAsync(&tableDev->entries[i], 0, sizeof(tableDev->entries[i]), stream),
                      ret, remove_winSorted);
        break;
      }
      if (tableHost->next == nullptr) break;
      tableDev = tableHost->next;
    }
  }

  // Free device window structure
  NCCLCHECKGOTO(ncclShadowPoolFree(&devr->shadows, winDev, stream), ret, remove_winSorted);

  // Deregister local handle
  NCCLCHECKGOTO(ncclCommDeregister(comm, winHost->localRegHandle), ret, remove_winSorted);

remove_winSorted:
  // Remove from sorted window list
  { int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted,
                              devr->winSortedCount, reinterpret_cast<uintptr_t>(winHost->userPtr));
    i -= 1;
    listRemove(devr->winSorted, &devr->winSortedCount, i);
  }
  free(winHost);
fail:
  return ret;
}
```

**Memory Drop Reference** (may unmap if last user):

**File**: [src/dev_runtime.cc:426-450](../../../thirdparty/nccl/src/dev_runtime.cc#L426)

```cpp
static void symMemoryDropRef(struct ncclComm* comm, struct ncclDevrMemory* mem) {
  if (mem != nullptr && 0 == --mem->refCount) {
    struct ncclDevrState* devr = &comm->devrState;

    // Deregister from GIN
    if (devr->ginEnabled) {
      ncclGinDeregister(comm, mem->ginHostWins);
    }

    // Unbind from all multicast teams
    for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
      symUnbindTeamMemory(comm, t, mem);
    }

    // Unmap from all LSA ranks
    for (int r = 0; r < devr->lsaSize; r++) {
      CUdeviceptr addr = reinterpret_cast<uintptr_t>((char*)devr->lsaFlatBase +
                                                      r*devr->bigSize + mem->bigOffset);
      CUCHECKIGNORE(cuMemUnmap(addr, mem->size));
    }

    // Free offset in big VA space
    ncclSpaceFree(&devr->bigSpace, mem->bigOffset, mem->size);

    // Release CUDA memory handle
    CUCHECKIGNORE(cuMemRelease(mem->memHandle));

    // Remove from memory list
    struct ncclDevrMemory** ptr = &devr->memHead;
    while (*ptr != mem) ptr = &(*ptr)->next;
    *ptr = mem->next;

    free(mem);
  }
}
```

---

## Symmetric Memory Deep Dive

### What "Symmetric" Actually Means

In NCCL's symmetric memory implementation, "symmetric" means:

1. **Consistent Virtual Address Layout**: All ranks map peer memories at predictable offsets in a shared virtual address space
2. **Same Offset Within Big VA Space**: A buffer registered at offset X occupies the same offset X in every rank's portion of the big VA space
3. **Direct Addressability**: Any rank can compute the address of any other rank's buffer using: `base + rank * stride + offset`

**NOT necessarily**:
- Same absolute virtual addresses across ranks (they can differ)
- Same physical memory locations
- Identical memory sizes

### Bootstrap/Exchange Protocol

The symmetric window registration follows this protocol:

```
Timeline of ncclCommWindowRegister:

Rank 0                  Rank 1                  Rank 2                  Rank 3
------                  ------                  ------                  ------
ncclCommWindowRegister called on all ranks
  ↓
ncclDevrInitOnce
  ↓
Enqueue task
  ↓
ncclGroupEnd triggered
  ↓
ncclDevrWindowRegisterInGroup
  ↓                       ↓                       ↓                       ↓
cuMemRetainAllocationHandle
  ↓                       ↓                       ↓                       ↓
symMemoryObtain
  ↓                       ↓                       ↓                       ↓
symMemoryMapLsaTeam
  ↓                       ↓                       ↓                       ↓
Export memory handle
  ↓                       ↓                       ↓                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│  bootstrapIntraNodeAllGather - exchange memory handles              │
│  Rank 0 → [handle0, ?, ?, ?]                                       │
│  Rank 1 → [?, handle1, ?, ?]                                       │
│  Rank 2 → [?, ?, handle2, ?]                                       │
│  Rank 3 → [?, ?, ?, handle3]                                       │
│                                                                      │
│  After AllGather:                                                   │
│  All ranks → [handle0, handle1, handle2, handle3]                  │
└─────────────────────────────────────────────────────────────────────┘
  ↓                       ↓                       ↓                       ↓
Reserve flat VA space (lsaFlatBase) if not exists
  ↓                       ↓                       ↓                       ↓
For each peer rank:
  Import handle
  cuMemMap(lsaFlatBase + rank*bigSize + offset)
  cuMemSetAccess
  ↓                       ↓                       ↓                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│  bootstrapIntraNodeBarrier - ensure all imports complete            │
└─────────────────────────────────────────────────────────────────────┘
  ↓                       ↓                       ↓                       ↓
symWindowCreate
  ↓                       ↓                       ↓                       ↓
Create device window structure
  ↓                       ↓                       ↓                       ↓
Add to lookup table
  ↓                       ↓                       ↓                       ↓
┌─────────────────────────────────────────────────────────────────────┐
│  bootstrapBarrier - world-wide synchronization                      │
└─────────────────────────────────────────────────────────────────────┘
  ↓                       ↓                       ↓                       ↓
Return ncclWindow_t
```

### Peer-to-Peer Memory Mapping Mechanics

**CUDA VMM APIs Used**:

1. **`cuMemGetAllocationGranularity`**: Determine alignment requirements
2. **`cuMemCreate`**: Allocate physical memory (done by `ncclMemAlloc`)
3. **`cuMemRetainAllocationHandle`**: Get shareable handle for existing allocation
4. **`cuMemExportToShareableHandle`**: Export for IPC sharing
5. **`cuMemImportFromShareableHandle`**: Import peer's handle
6. **`cuMemAddressReserve`**: Reserve VA space for flat mapping
7. **`cuMemMap`**: Map physical memory to VA range
8. **`cuMemSetAccess`**: Set access permissions
9. **`cuMemUnmap`**: Unmap during cleanup
10. **`cuMemAddressFree`**: Free VA space
11. **`cuMemRelease`**: Release handle reference

**Mapping Sequence**:

```cpp
// Each rank does this for every peer in LSA team:

// 1. Reserve large contiguous VA space (once)
CUdeviceptr flatBase;
cuMemAddressReserve(&flatBase, lsaSize * bigSize, alignment, 0, 0);
// Creates VA space: [flatBase ... flatBase + lsaSize*bigSize)

// 2. For each peer rank r:
CUmemGenericAllocationHandle peerHandle = /* received via AllGather */;
CUdeviceptr targetAddr = flatBase + r * bigSize + bufferOffset;

// 3. Map peer's physical memory to target VA
cuMemMap(targetAddr, bufferSize, 0, peerHandle, 0);

// 4. Set read/write permissions
CUmemAccessDesc access = {
  .location = {.type = CU_MEM_LOCATION_TYPE_DEVICE, .id = cudaDev},
  .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
};
cuMemSetAccess(targetAddr, bufferSize, &access, 1);
```

**Result**: All ranks can now access each other's buffers through the flat VA space.

### Address Translation

**On Device** (in kernel):

```cuda
// Window structure on device
struct ncclWindow_vidmem {
  char* lsaFlatBase;    // Points to rank 0's buffer start
  uint32_t stride4G;    // Stride between ranks in 4GB units
  uint32_t mcOffset4K;  // Multicast offset in 4KB units
  int lsaRank;          // My rank in LSA team
  ...
};

// To access rank r's buffer at element offset elemOffset:
template<typename T>
__device__ T* getPeerPtr(ncclWindow_vidmem* win, int peerRank, size_t elemOffset) {
  size_t stride = (size_t)win->stride4G << 32;  // Convert 4GB units to bytes
  return (T*)(win->lsaFlatBase + peerRank * stride) + elemOffset;
}
```

**No address translation needed!** The symmetric VA layout means direct pointer arithmetic.

---

## Performance Optimizations

### What Algorithms Are Enabled by Symmetric Memory

1. **Direct Peer Reads/Writes**: No intermediate buffers or copies
2. **Ring AllReduce Without Send/Recv**: Each thread directly reads from next rank
3. **Multicast Operations**: NVLS support for single write → multiple reads
4. **Pipelined Reductions**: Overlapping fetch from multiple peers with reduction

### Kernel Implementation Differences

**Regular NCCL Kernel** (simplified):

```cuda
// Traditional approach: send/recv through channels
__device__ void allReduceRing(Args args) {
  // Send my data to next rank
  ncclSend(sendbuff, count, peer);

  // Receive from previous rank
  ncclRecv(recvbuff, count, peer);

  // Reduce received data
  reduce(recvbuff, sendbuff, count);

  // Repeat for N-1 steps...
}
```

**Symmetric Kernel** (from `src/device/symmetric/all_reduce.cuh`):

```cuda
// Direct peer access - no send/recv
__device__ void allReduceSymmetric(ncclSymPtr input, ncclSymPtr output) {
  // Directly read from ALL peers in parallel
  for (int r = 0; r < nRanks; r++) {
    T data = input.peerPtr(world, r)[myOffset];  // Direct load from peer r
    acc = reduce(acc, data);
  }

  // Write result to ALL peers in parallel
  for (int r = 0; r < nRanks; r++) {
    output.peerPtr(world, r)[myOffset] = acc;  // Direct store to peer r
  }
}
```

### Latency/Bandwidth Improvements

From the performance model in [src/sym_kernels.cc:117-120](../../../thirdparty/nccl/src/sym_kernels.cc#L117):

```cpp
static double model(double busBytes, double baseLat, int nSMs, double smBw,
                    double busMultiplier, double peakBw) {
  double bw = softmin(nSMs*smBw*busMultiplier, peakBw, smBw);
  return baseLat + softplus(busBytes/bw - 1, 1);
}
```

**Parameters for AllReduce on Ampere vs Hopper**:

```cpp
if (comm->cudaArch < 1000) {  // Ampere
  baseLat = isLL ? 4.5 : 7.8;
  smBw = isAR ? 65*GBps : 44*GBps;
  peakBw = k == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 480*GBps : 320*GBps;
} else {  // Hopper
  baseLat = isLL ? (isAG ? 8.5 : 11) : (isAR ? 19.5 : 13.0);
  smBw = 55*GBps;
  peakBw = k == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 1000*GBps : 600*GBps;
}
```

**Multicast Bandwidth**: Up to **1000 GB/s** on Hopper with NVLS!

**Improvements Over Regular AllReduce**:
- **Latency**: Lower base latency (direct access vs send/recv overhead)
- **Bandwidth**: 2-3x higher with multicast on supported hardware
- **Scalability**: Better scaling with rank count (direct access vs round-robin)

---

## Data Structures

### ncclWindow_vidmem (Device-Side)

**File**: [src/include/nccl_device/impl/core__types.h:13-22](../../../thirdparty/nccl/src/include/nccl_device/impl/core__types.h#L13)

```cpp
struct ncclWindow_vidmem {
  void* winHost;                              // Backpointer to ncclDevrWindow
  char* lsaFlatBase;                          // Points to rank 0's buffer in flat VA space
  int lsaRank;                                // This rank's index in LSA team
  int worldRank;                              // This rank's world rank
  uint32_t stride4G;                          // Stride between ranks (in 4GB units)
  uint32_t mcOffset4K;                        // Multicast offset (in 4KB units)
  uint32_t ginOffset4K;                       // GIN (GPU Initiated Network) offset
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONTEXTS];  // GIN window handles
};
```

**Field Explanations**:

- **`lsaFlatBase`**: Base address for rank 0's buffer. Rank r's address = `lsaFlatBase + r * (stride4G << 32)`
- **`stride4G`**: Stored in 4GB units to fit in 32 bits. Full stride = `(size_t)stride4G << 32`
- **`mcOffset4K`**: For multicast operations, offset within multicast region
- **`ginWins`**: For GPU-initiated network operations (multi-node)

### ncclDevrWindow (Host-Side)

**File**: [src/include/dev_runtime.h:20-28](../../../thirdparty/nccl/src/include/dev_runtime.h#L20)

```cpp
struct ncclDevrWindow {
  struct ncclDevrMemory* memory;    // Underlying memory allocation
  void* userPtr;                    // User's original buffer pointer
  size_t size;                      // Window size in bytes
  size_t bigOffset;                 // Offset within big VA space
  int winFlags;                     // NCCL_WIN_COLL_SYMMETRIC, etc.
  void* localRegHandle;             // Local registration handle (IPC/RDMA)
  struct ncclWindow_vidmem* vidmem; // Device-side window structure
};
```

### ncclDevrMemory

**File**: [src/dev_runtime.cc:17-26](../../../thirdparty/nccl/src/dev_runtime.cc#L17)

```cpp
struct ncclDevrMemory {
  int refCount;                               // Number of windows using this memory
  struct ncclDevrMemory* next;                // Linked list of all registered memories
  CUmemGenericAllocationHandle memHandle;     // CUDA VMM handle
  void* primaryAddr;                          // Primary VA (may be user's or flat VA)
  size_t size;                                // Memory size
  size_t bigOffset;                           // Offset in big VA space
  void* ginHostWins[NCCL_GIN_MAX_CONTEXTS];   // GIN host-side windows
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONTEXTS];  // GIN device-side windows
};
```

**Reference Counting**: Multiple windows can share the same underlying memory allocation.

### ncclDevrState

**File**: [src/include/dev_runtime.h:46-68](../../../thirdparty/nccl/src/include/dev_runtime.h#L46)

```cpp
struct ncclDevrState {
  // LSA (Local Symmetric Access) team information
  int lsaSelf;                      // My rank in LSA team
  int lsaSize;                      // Number of ranks in LSA team
  int* lsaRankList;                 // Array of world ranks in LSA team

  size_t granularity;               // CUDA VMM allocation granularity
  bool ginEnabled;                  // GPU Initiated Network enabled?

  struct ncclDevrMemory* memHead;   // Linked list of registered memories

  struct ncclDevrWindowSorted* winSorted;  // Sorted array for fast window lookup
  int winSortedCapacity, winSortedCount;

  struct ncclDevrTeam* teamHead;    // Linked list of multicast teams

  size_t bigSize;                   // Size of big VA space per rank
  struct ncclSpace bigSpace;        // Allocator for offsets in big VA
  void* lsaFlatBase;                // Base of flat VA space (size = lsaSize * bigSize)

  struct ncclShadowPool shadows;    // Pool for host/device shadow structures
  struct ncclDevCommWindowTable* windowTable;  // Device-side window lookup table

  // Task queues for deferred operations
  struct ncclIntruQueue<struct ncclDevrRegTask, &ncclDevrRegTask::next> regTaskQueue;
  struct ncclIntruQueue<struct ncclDevrCommCreateTask, &ncclDevrCommCreateTask::next> commCreateTaskQueue;
};
```

---

## Sequence Diagrams

### Symmetric Window Registration

```
User Thread                    NCCL Host                   Bootstrap               Device
     │                              │                          │                      │
     ├─ ncclCommWindowRegister ────>│                          │                      │
     │                              ├─ ncclGroupStart          │                      │
     │                              ├─ ncclDevrInitOnce        │                      │
     │                              │  ├─ Calculate LSA team   │                      │
     │                              │  ├─ Reserve big VA space │                      │
     │                              │  └─ Initialize allocators│                      │
     │                              ├─ Enqueue reg task        │                      │
     │                              └─ ncclGroupCommJoin       │                      │
     ├─ ncclGroupEnd ──────────────>│                          │                      │
     │                              ├─ Execute reg task        │                      │
     │                              ├─ ncclCommRegister (local)│                      │
     │                              ├─ cuMemRetainAllocationHandle                    │
     │                              ├─ cuMemExportToShareableHandle                   │
     │                              │                          │                      │
     │                              ├─ AllGather handles ─────>│                      │
     │                              │<─────────────────────────┤                      │
     │                              │  [handle0, handle1, ...]  │                      │
     │                              │                          │                      │
     │                              ├─ For each peer:          │                      │
     │                              │  ├─ cuMemImportFromShareableHandle              │
     │                              │  ├─ cuMemMap(flat VA + rank*bigSize + offset)   │
     │                              │  └─ cuMemSetAccess       │                      │
     │                              │                          │                      │
     │                              ├─ Barrier ────────────────>│                      │
     │                              │<─────────────────────────┤                      │
     │                              │                          │                      │
     │                              ├─ Create ncclWindow_vidmem                       │
     │                              ├─ cudaMemcpyAsync ────────┼──────────────────────>│
     │                              ├─ Add to lookup table     │                      │
     │                              │                          │                      │
     │                              ├─ World barrier ──────────>│                      │
     │                              │<─────────────────────────┤                      │
     │<─ Return ncclWindow_t ───────┤                          │                      │
```

### AllReduce with Symmetric Windows

```
User Thread              NCCL Enqueue             Kernel Selection            Device Kernel
     │                        │                          │                          │
     ├─ ncclAllReduce ───────>│                          │                          │
     │                        ├─ Find windows            │                          │
     │                        ├─ Check SYMMETRIC flag    │                          │
     │                        ├─ ncclSymkAvailable ─────>│                          │
     │                        │<─ true ──────────────────┤                          │
     │                        ├─ ncclSymkPickKernel ────>│                          │
     │                        │                          ├─ Evaluate candidates     │
     │                        │                          ├─ Performance model       │
     │                        │<─ Best kernel ID ────────┤                          │
     │                        ├─ Prepare args            │                          │
     │                        ├─ Launch kernel ──────────┼──────────────────────────>│
     │                        │                          │                          ├─ Load args
     │                        │                          │                          ├─ Get window
     │                        │                          │                          ├─ For each peer:
     │                        │                          │                          │  ├─ Calculate address
     │                        │                          │                          │  │  addr = lsaFlatBase
     │                        │                          │                          │  │    + peer * stride4G<<32
     │                        │                          │                          │  ├─ Direct load
     │                        │                          │                          │  └─ Reduce
     │                        │                          │                          ├─ For each peer:
     │                        │                          │                          │  ├─ Direct store result
     │                        │                          │                          │
     │                        │<─────────────────────────┼──────────────────────────┤ Kernel complete
     ├─ cudaStreamSynchronize │                          │                          │
     │<───────────────────────┤                          │                          │
```

### Window Deregistration

```
User Thread                    NCCL Host                   Device
     │                              │                          │
     ├─ ncclCommWindowDeregister ──>│                          │
     │                              ├─ Get window structure    │
     │                              ├─ Remove from table       │
     │                              ├─ cudaMemsetAsync ────────>│ Clear entry
     │                              ├─ Free device window      │
     │                              ├─ symMemoryDropRef        │
     │                              │  ├─ Decrement refCount   │
     │                              │  └─ If refCount == 0:    │
     │                              │     ├─ For each LSA rank:│
     │                              │     │  └─ cuMemUnmap     │
     │                              │     ├─ ncclSpaceFree     │
     │                              │     └─ cuMemRelease      │
     │                              ├─ ncclCommDeregister      │
     │                              │   (local handle)         │
     │<─ Return ─────────────────────┤                          │
```

---

## Summary: Key Differences from Example 04

| Aspect | Example 04 (Buffer Registration) | Example 05 (Symmetric Memory) |
|--------|-----------------------------------|-------------------------------|
| **API** | `ncclCommRegister` | `ncclCommWindowRegister` with `NCCL_WIN_COLL_SYMMETRIC` |
| **Purpose** | Enable IPC/RDMA for P2P transfers | Create unified symmetric VA space for direct access |
| **Collective** | No (local only) | Yes (requires bootstrap exchange and barriers) |
| **Memory Mapping** | OS/driver handles P2P | Explicit CUDA VMM mapping by NCCL |
| **VA Layout** | Independent per rank | Coordinated flat VA space |
| **Kernel Type** | Standard NCCL primitives | Specialized symmetric kernels |
| **Peer Access** | Through send/recv channels | Direct load/store via `peerPtr()` |
| **Multicast** | No | Yes (NVLS on Hopper+) |
| **Performance** | Good | Better (lower latency, higher BW) |
| **Complexity** | Low | High (VMM, bootstrap, barriers) |

**When to Use Symmetric Memory**:
- ✅ Single-node or intra-node collectives
- ✅ Large buffers that benefit from multicast
- ✅ Hopper+ GPUs with NVLS support
- ✅ Applications that can pre-register buffers
- ❌ Small messages (overhead not worth it)
- ❌ Dynamic/unpredictable buffer patterns
- ❌ Multi-node (limited benefit without NVLS)

---

## References

**Source Files**:
- Example: [examples/05_symmetric_memory/01_allreduce/main.cc](../../../thirdparty/nccl/examples/05_symmetric_memory/01_allreduce/main.cc)
- Public API: [src/nccl.h.in](../../../thirdparty/nccl/src/nccl.h.in)
- Runtime implementation: [src/dev_runtime.cc](../../../thirdparty/nccl/src/dev_runtime.cc)
- Kernel selection: [src/sym_kernels.cc](../../../thirdparty/nccl/src/sym_kernels.cc)
- AllReduce kernel: [src/device/symmetric/all_reduce.cuh](../../../thirdparty/nccl/src/device/symmetric/all_reduce.cuh)
- Primitives: [src/device/symmetric/primitives.cuh](../../../thirdparty/nccl/src/device/symmetric/primitives.cuh)
- Data structures: [src/include/dev_runtime.h](../../../thirdparty/nccl/src/include/dev_runtime.h), [src/include/sym_kernels.h](../../../thirdparty/nccl/src/include/sym_kernels.h), [src/include/nccl_device/impl/core__types.h](../../../thirdparty/nccl/src/include/nccl_device/impl/core__types.h)

**Related Documentation**:
- CUDA Virtual Memory Management: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
- NVLS (NVLink Sharp): https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html
