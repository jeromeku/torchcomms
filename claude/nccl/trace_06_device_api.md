# NCCL Device API - Complete Execution Trace

Complete literate code walkthrough of NCCL's Device API using Example 06.

**Example Source**: [/home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/01_allreduce/main.cu](/home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/01_allreduce/main.cu)

---

## Table of Contents

1. [Overview](#overview)
2. [What is the Device API?](#what-is-the-device-api)
3. [Architecture Overview](#architecture-overview)
4. [Complete Execution Flow](#complete-execution-flow)
5. [Host-Side Setup Traces](#host-side-setup-traces)
6. [Device-Side Execution Traces](#device-side-execution-traces)
7. [LSA Barrier Deep Dive](#lsa-barrier-deep-dive)
8. [Device Communicator Structure](#device-communicator-structure)
9. [Comparison with Host API](#comparison-with-host-api)
10. [Data Structures](#data-structures)
11. [Sequence Diagrams](#sequence-diagrams)
12. [Performance Characteristics](#performance-characteristics)

---

## Overview

### What is the Device API?

The **NCCL Device API** is a set of functions and data structures that enable GPU kernels to perform collective communication operations **directly from device code**, without requiring CPU intervention. This is fundamentally different from NCCL's traditional Host API, where the CPU initiates all communication operations.

**Key Characteristics:**

1. **Kernel-Initiated Communication**: GPU threads can trigger and participate in collective operations from within CUDA kernels
2. **Direct Peer Memory Access**: Device code can read/write remote GPU memory using LSA (Load/Store Accessible) pointers
3. **Cross-GPU Synchronization**: LSA barriers enable coordination between threads across different GPUs
4. **Reduced Latency**: Eliminates host-device round trips for communication-heavy kernels
5. **Computation-Communication Fusion**: Allows seamless interleaving of compute and communication in a single kernel

### Key Use Cases

1. **Fusion Opportunities**: Combine computation and communication in a single kernel launch to reduce scheduling overhead
2. **Reduced Latency**: Eliminate host-device synchronization for small, frequent communication patterns
3. **Custom Collectives**: Implement application-specific collective operations not available in standard NCCL
4. **Iterative Algorithms**: Enable GPU-resident iterative algorithms that need inter-GPU communication per iteration
5. **Fine-Grained Synchronization**: Control communication at CTA or warp granularity

### Architecture Overview

The Device API consists of two main execution phases:

#### Host-Side Setup
- Initialize standard NCCL communicator
- Register symmetric memory windows for peer access
- Create device communicator with LSA barrier allocation
- Pass device communicator to kernel

#### Device-Side Execution
- Launch user kernel with device communicator
- Use LSA barriers for cross-GPU synchronization
- Access peer memory via `ncclGetLsaPointer`
- Perform custom collective logic in device code

---

## What is the Device API?

### Device API vs Host API

| Aspect | Host API | Device API |
|--------|----------|------------|
| **Initiation** | CPU calls `ncclAllReduce`, etc. | GPU kernel performs communication directly |
| **Synchronization** | CUDA streams and events | LSA barriers across GPUs |
| **Memory Access** | NCCL internal kernels handle data movement | User kernel directly reads/writes peer memory |
| **Latency** | Host-device synchronization overhead | Lower latency for small operations |
| **Flexibility** | Fixed collective operations | Custom communication patterns possible |
| **Complexity** | Simple API, NCCL handles details | User manages synchronization and memory access |
| **Typical Use** | Standard collectives (AllReduce, AllGather, etc.) | Fused compute-communication, custom patterns |

### When to Use Device API

**Use Device API when:**
- Compute kernels need immediate communication results within the same kernel
- Communication patterns are fine-grained and frequent (high host overhead in Host API)
- Implementing custom collective operations not available in standard NCCL
- Overlapping computation and communication within a single kernel is beneficial
- Latency is critical and host-device synchronization is a bottleneck

**Use Host API when:**
- Standard collective operations are sufficient
- Communication and computation are naturally separated
- Simplicity and ease of use are priorities
- Leveraging NCCL's optimized collective algorithms is important
- Dealing with large data transfers where host overhead is amortized

---

## Complete Execution Flow

### High-Level Call Flow

```
User Application (main.cu)
│
├─> [HOST SETUP PHASE]
│   │
│   ├─> ncclGetUniqueId()                    [Standard NCCL init]
│   ├─> ncclCommInitRank()                   [Create communicator]
│   │
│   ├─> ncclMemAlloc()                       [Allocate device buffers]
│   │   └─> CUDA VMM allocation
│   │
│   ├─> ncclCommWindowRegister()             [Register symmetric windows]
│   │   ├─> ncclDevrWindowRegisterInGroup()
│   │   ├─> symMemoryObtain()                [Exchange memory handles]
│   │   ├─> symMemoryMapLsaTeam()            [Map to flat VA space]
│   │   └─> symWindowCreate()                [Create window structure]
│   │
│   └─> ncclDevCommCreate()                  [Create device communicator]
│       ├─> ncclDevrCommCreateInternal()
│       ├─> symTeamObtain()                  [Get LSA team]
│       ├─> ncclLsaBarrierCreateRequirement() [Allocate barriers]
│       └─> Allocate resource buffer         [For barrier state]
│
├─> [DEVICE EXECUTION PHASE]
│   │
│   └─> simpleAllReduceKernel<<<>>>()        [Launch user kernel]
│       │
│       ├─> ncclLsaBarrierSession constructor [Initialize barrier]
│       │   └─> Load epoch from barrier state
│       │
│       ├─> bar.sync(memory_order_relaxed)   [Entry barrier]
│       │   ├─> arrive()                     [Signal peers]
│       │   │   └─> atomic store to peer inboxes
│       │   └─> wait()                       [Wait for peers]
│       │       └─> atomic load from local inbox
│       │
│       ├─> [COMPUTATION LOOP]
│       │   │
│       │   ├─> ncclGetLsaPointer()          [Get peer memory pointer]
│       │   │   └─> Calculate: lsaFlatBase + peer*stride4G + offset
│       │   │
│       │   ├─> Load from peer memory        [Direct RDMA read]
│       │   ├─> Local reduction              [Accumulate values]
│       │   │
│       │   ├─> ncclGetLsaPointer()          [Get output pointer]
│       │   └─> Store to all peer memories   [Direct RDMA write]
│       │
│       ├─> bar.sync(memory_order_release)   [Exit barrier]
│       │   └─> Ensure all writes complete before unblocking
│       │
│       └─> ~ncclLsaBarrierSession()         [Destructor saves epoch]
│
└─> [CLEANUP PHASE]
    ├─> ncclDevCommDestroy()
    ├─> ncclCommWindowDeregister()
    ├─> ncclMemFree()
    └─> ncclCommDestroy()
```

---

## Host-Side Setup Traces

### 5.1 Standard NCCL Initialization

The Device API builds on top of standard NCCL communicator initialization.

**File:** [main.cu:113-136](/home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/01_allreduce/main.cu#L113)

```cpp
// Step 1: Get unique ID on rank 0
if (my_rank == 0) {
  NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
}

// Step 2: Broadcast unique ID to all ranks
util_broadcast(0, my_rank, &nccl_unique_id);

// Step 3: Set CUDA device for this rank
CUDACHECK(cudaSetDevice(local_device));

// Step 4: Initialize NCCL communicator
// This is identical to Host API - creates standard ncclComm_t
NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_unique_id, my_rank));
```

**Key Point:** Device API requires the same communicator initialization as Host API. The `ncclComm_t` is shared between both APIs.

---

### 5.2 Memory Allocation with ncclMemAlloc

Device API requires memory allocated with `ncclMemAlloc` to ensure CUDA VMM support.

**File:** [main.cu:138-151](/home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/01_allreduce/main.cu#L138)

```cpp
// Allocate memory for AllReduce operation
size_t count = 1024 * 1024; // 1M elements
size_t size_bytes = count * sizeof(float);

float *h_data = (float*)malloc(size_bytes);
void* d_sendbuff;
void* d_recvbuff;
ncclWindow_t send_win;
ncclWindow_t recv_win;

// Device API requires allocation compatible with symmetric memory
// This uses CUDA VMM to enable cross-GPU memory handle sharing
NCCLCHECK(ncclMemAlloc(&d_sendbuff, size_bytes));
NCCLCHECK(ncclMemAlloc(&d_recvbuff, size_bytes));
```

**Why `ncclMemAlloc` is required:**
- Uses CUDA Virtual Memory Management (VMM) APIs (`cuMemCreate`, `cuMemMap`)
- Allocates memory with shareable handles (POSIX FD or FABRIC handles)
- Enables memory to be mapped into symmetric VA space across GPUs
- Supports optional fabric handles for NVLS/NVSwitch

**Implementation:** See [src/allocator.cc:12-98](/home/jeromeku/torchcomms/thirdparty/nccl/src/allocator.cc#L12) for full `ncclMemAlloc` implementation.

---

### 5.3 Window Registration: ncclCommWindowRegister

This is the critical step that enables device-side peer memory access.

#### User Code

**File:** [main.cu:154-160](/home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/01_allreduce/main.cu#L154)

```cpp
// Register symmetric windows for LSA (Load/Store Accessible) access
// Windows enable direct peer-to-peer access from device kernels
NCCLCHECK(ncclCommWindowRegister(comm, d_sendbuff, size_bytes, &send_win,
                                 NCCL_WIN_COLL_SYMMETRIC));
NCCLCHECK(ncclCommWindowRegister(comm, d_recvbuff, size_bytes, &recv_win,
                                 NCCL_WIN_COLL_SYMMETRIC));
```

**What happens:**
1. Each rank registers its local buffer
2. NCCL exchanges memory handles across the LSA team (local GPUs)
3. Each rank maps all peers' memory into a flat, symmetric VA space
4. Returns `ncclWindow_t` handle containing metadata for device access

#### Implementation: ncclCommWindowRegister Entry Point

**File:** [src/dev_runtime.cc:890-924](/home/jeromeku/torchcomms/thirdparty/nccl/src/dev_runtime.cc#L890)

```cpp
NCCL_API(ncclResult_t, ncclCommWindowRegister,
         ncclComm_t comm, void* ptr, size_t size,
         ncclWindow_t* win, int winFlags);

ncclResult_t ncclCommWindowRegister(
    struct ncclComm* comm, void* userPtr, size_t userSize,
    struct ncclWindow_vidmem** outWinDev, int winFlags
  ) {
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  struct ncclDevrRegTask* task;

  // Save current device and start group operation
  CUDACHECK(cudaGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());

  // Validate parameters
  if (userPtr == nullptr || userSize == 0 ||
      !(comm->symmetricSupport || ncclParamLocalRegister()))
    goto exit;

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  // Initialize device runtime state (LSA team, VA space, etc.)
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  // Create registration task and enqueue it
  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);
  task->userPtr = userPtr;
  task->userSize = userSize;
  task->winFlags = winFlags;
  task->outWinDev = outWinDev;
  ncclIntruQueueEnqueue(&comm->devrState.regTaskQueue, task);

  // Join the group for coordinated execution across ranks
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  cudaSetDevice(saveDev);
  return ret;
fail:
  goto exit;
}
```

**Key Points:**
- Window registration is a collective operation (requires coordination across ranks)
- Uses NCCL group semantics to batch multiple registrations
- Defers actual work to `ncclDevrWindowRegisterInGroup` during group finalization

#### ncclDevrWindowRegisterInGroup - The Core Implementation

**File:** [src/dev_runtime.cc:578-648](/home/jeromeku/torchcomms/thirdparty/nccl/src/dev_runtime.cc#L578)

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

  // Step 1: Register buffer locally for IPC
  NCCLCHECKGOTO(ncclCommRegister(comm, userPtr, userSize, &localRegHandle),
                ret, fail);

  // Check if symmetric support is available
  if (!comm->symmetricSupport) {
    // Fall back to local registration only
    *outWinDev = reinterpret_cast<struct ncclWindow_vidmem*>(localRegHandle);
    return ncclSuccess;
  }

  // Step 2: Enable symmetric kernels if this is the first symmetric window
  if (winFlags & NCCL_WIN_COLL_SYMMETRIC) {
    NCCLCHECKGOTO(ncclSymkInitOnce(comm), ret, fail);
  }

  // Step 3: Get underlying CUDA memory handle
  // Query the VA range containing userPtr
  CUCHECKGOTO(cuMemGetAddressRange(&memAddr, &memSize,
              reinterpret_cast<CUdeviceptr>(userPtr)), ret, fail_locReg);
  memOffset = reinterpret_cast<CUdeviceptr>(userPtr) - memAddr;

  // Validate alignment (required for symmetric windows)
  if (memOffset % NCCL_WIN_REQUIRED_ALIGNMENT != 0) {
    WARN("Window address must be suitably aligned.");
    ret = ncclInvalidArgument;
    goto fail;
  }

  // Get shareable memory handle
  CUCHECKGOTO(cuMemRetainAllocationHandle(&memHandle,
              reinterpret_cast<void*>(memAddr)), ret, fail_locReg);

  // Step 4: Trade memory handle for ncclDevrMemory structure
  // This exchanges handles across LSA team and maps to flat VA space
  NCCLCHECKGOTO(symMemoryObtain(comm, memHandle, (void*)memAddr, memSize, &mem),
                ret, fail_locReg_memHandle);
  memHandle = 0x0; // symMemoryObtain took our reference

  // Step 5: Create CUDA stream for async operations
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                ret, fail);

  // Step 6: Create window structure with device-side metadata
  NCCLCHECKGOTO(symWindowCreate(
      comm, mem, memOffset, userPtr, userSize, winFlags, localRegHandle,
      outWinDev, nullptr, stream
    ), ret, fail_locReg_memHandle_mem_stream);
  mem = nullptr; // symWindowCreate took our reference

  // Step 7: Synchronize stream to ensure device copies complete
  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret,
                fail_locReg_memHandle_mem_stream_win);

  // Step 8: Barrier across all ranks
  // Required because memory mapping and window creation must complete globally
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef),
                ret, fail_locReg_memHandle_mem_stream_win);

  cudaStreamDestroy(stream);
  return ret;

// Error handling omitted for brevity
}
```

#### symMemoryObtain - Exchange Memory Handles

**File:** [src/dev_runtime.cc:360-424](/home/jeromeku/torchcomms/thirdparty/nccl/src/dev_runtime.cc#L360)

```cpp
// On success we take caller's reference on memHandle.
// Due to multicast binds for each pre-existing team, this function requires
// caller do a world barrier before returning to user.
static ncclResult_t symMemoryObtain(
    struct ncclComm* comm, CUmemGenericAllocationHandle memHandle,
    void* memAddr, size_t size,
    struct ncclDevrMemory** outMem
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  int64_t bigOffset = 0;

  // Check if this memory handle is already registered
  struct ncclDevrMemory* mem = devr->memHead;
  while (mem != nullptr) {
    if (mem->memHandle == memHandle) {
      CUCHECKIGNORE(cuMemRelease(memHandle));
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

  // Step 1: Allocate offset in the "big" symmetric VA space
  // Each rank has a 128GB (or larger) VA reservation
  NCCLCHECKGOTO(ncclSpaceAlloc(&devr->bigSpace, devr->bigSize, size,
                devr->granularity, &bigOffset), ret, fail_mem);
  mem->bigOffset = bigOffset;

  // Step 2: Map memory into flat LSA VA space for all peers
  // This is the critical step that creates symmetric addressing
  NCCLCHECKGOTO(symMemoryMapLsaTeam(comm, memHandle, size, bigOffset),
                ret, fail_mem_space);

  // Step 3: If caller doesn't have a VA, use the LSA mapping
  if (mem->primaryAddr == nullptr) {
    mem->primaryAddr = (char*)devr->lsaFlatBase + devr->lsaSelf*devr->bigSize
                       + mem->bigOffset;
  }

  // Step 4: Bind new memory with each existing multicast team (if any)
  for (struct ncclDevrTeam* t = devr->teamHead; t != nullptr; t = t->next) {
    NCCLCHECKGOTO(symBindTeamMemory(comm, t, mem), ret, fail_mem_space_teams);
  }

  // Step 5: Register with GIN if enabled
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

// Error handling omitted
}
```

#### symMemoryMapLsaTeam - Create Flat Symmetric VA Space

**File:** [src/dev_runtime.cc:147-203](/home/jeromeku/torchcomms/thirdparty/nccl/src/dev_runtime.cc#L147)

```cpp
static ncclResult_t symMemoryMapLsaTeam(
    struct ncclComm* comm, CUmemGenericAllocationHandle memHandle,
    size_t size, size_t bigOffset
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  CUmemAccessDesc accessDesc = {};

  // Message type for handle exchange
  union Message {
    CUmemGenericAllocationHandle memHandle;
    CUmemFabricHandle fabricHandle;
  };

  // Allocate message buffer for LSA team exchange
  Message* messages = (Message*)calloc(devr->lsaSize, sizeof(Message));

  // Step 1: Export our memory handle
  if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    messages[devr->lsaSelf].memHandle = memHandle;
  } else {
    // Export as fabric handle for NVLS
    CUCHECKGOTO(cuMemExportToShareableHandle(&messages[devr->lsaSelf].fabricHandle,
                memHandle, ncclCuMemHandleType, 0), ret, fail);
  }

  // Step 2: AllGather memory handles across LSA team
  // Uses bootstrap network for intra-node communication
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, devr->lsaRankList,
                devr->lsaSelf, devr->lsaSize, messages, sizeof(Message)),
                ret, fail);

  // Step 3: Create flat VA space on first use
  if (devr->lsaFlatBase == nullptr) {
    CUdeviceptr addr;
    // Reserve contiguous VA space: lsaSize * bigSize (e.g., 8 GPUs * 128GB)
    CUCHECKGOTO(cuMemAddressReserve(&addr, devr->lsaSize * devr->bigSize,
                NCCL_MAX_PAGE_SIZE, 0, 0), ret, fail);
    devr->lsaFlatBase = reinterpret_cast<void*>(addr);
  }

  // Step 4: Setup memory access permissions
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = comm->cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  // Step 5: Map each peer's memory into our flat VA space
  for (int r = 0; r < devr->lsaSize; r++) {
    CUmemGenericAllocationHandle impHandle;

    if (r == devr->lsaSelf) {
      // Use our own handle directly
      impHandle = memHandle;
    } else {
      // Import peer's handle
      if (ncclCuMemHandleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        int fd = -1;
        // Get file descriptor from proxy
        NCCLCHECKGOTO(ncclProxyClientGetFdBlocking(comm, devr->lsaRankList[r],
                      &messages[r], &fd), ret, fail);
        CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle,
                    reinterpret_cast<void*>((uintptr_t)fd),
                    ncclCuMemHandleType), ret, fail);
        SYSCHECKGOTO(close(fd), "close", ret, fail);
      } else {
        // Import fabric handle
        CUCHECKGOTO(cuMemImportFromShareableHandle(&impHandle,
                    (void*)&messages[r].fabricHandle,
                    ncclCuMemHandleType), ret, fail);
      }
    }

    // Calculate VA for this peer's memory
    // Layout: [rank0 bigSize][rank1 bigSize]...[rankN bigSize]
    //         Each rank's memory at offset bigOffset within their slice
    CUdeviceptr addr = reinterpret_cast<uintptr_t>(
      (char*)devr->lsaFlatBase + r*devr->bigSize + bigOffset
    );

    // Map the memory into VA space
    CUCHECKGOTO(cuMemMap(addr, size, 0, impHandle, 0), ret, fail);
    CUCHECKGOTO(cuMemSetAccess(addr, size, &accessDesc, 1), ret, fail);

    // Release imported handle (we keep a reference via mapping)
    if (r != devr->lsaSelf) {
      CUCHECKGOTO(cuMemRelease(impHandle), ret, fail);
    }
  }

  // Step 6: Barrier to ensure all ranks have imported our handle
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, devr->lsaRankList,
                devr->lsaSelf, devr->lsaSize, 0xbeef), ret, fail);
leave:
  free(messages);
  return ret;
fail:
  goto leave;
}
```

**Critical Insight - Symmetric VA Layout:**

After `symMemoryMapLsaTeam`, each rank has a flat VA space where:

```
Rank 0's view:
  [0x100000000000 .. 0x100020000000) -> Rank 0's memory (local)
  [0x100020000000 .. 0x100040000000) -> Rank 1's memory (remote)
  [0x100040000000 .. 0x100060000000) -> Rank 2's memory (remote)
  ...

Rank 1's view:
  [0x100000000000 .. 0x100020000000) -> Rank 0's memory (remote)
  [0x100020000000 .. 0x100040000000) -> Rank 1's memory (local)
  [0x100040000000 .. 0x100060000000) -> Rank 2's memory (remote)
  ...
```

**All ranks have the same VA layout, just different local/remote designations.**

#### symWindowCreate - Device-Side Window Metadata

**File:** [src/dev_runtime.cc:463-536](/home/jeromeku/torchcomms/thirdparty/nccl/src/dev_runtime.cc#L463)

```cpp
static ncclResult_t symWindowCreate(
    struct ncclComm* comm, struct ncclDevrMemory* mem,
    size_t memOffset, void* userPtr, size_t userSize, int winFlags,
    void* localReg,
    struct ncclWindow_vidmem** outWinDev, struct ncclDevrWindow** outWin,
    cudaStream_t stream
  ) {
  uintptr_t userAddr = reinterpret_cast<uintptr_t>(userPtr);
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclDevrWindow* win;

  // Step 1: Create host-side window tracking structure
  win = (struct ncclDevrWindow*)malloc(sizeof(struct ncclDevrWindow));
  memset(win, 0, sizeof(*win));
  win->memory = mem;
  win->size = userSize;
  win->bigOffset = mem->bigOffset + memOffset;
  win->winFlags = winFlags;
  win->localRegHandle = localReg;

  // Determine user-visible pointer
  if (userPtr == nullptr) {
    // No VA provided - use LSA flat VA address
    win->userPtr = (char*)devr->lsaFlatBase + (devr->lsaSelf*devr->bigSize)
                   + mem->bigOffset;
  } else {
    win->userPtr = userPtr;
  }

  // Step 2: Allocate device-side window structure
  struct ncclWindow_vidmem* winDev;
  struct ncclWindow_vidmem* winDevHost;
  NCCLCHECK(ncclShadowPoolAlloc(&devr->shadows, &winDev, &winDevHost, stream));

  win->vidmem = winDev;

  // Step 3: Populate device-side window metadata
  // This structure will be accessed by device code via ncclGetLsaPointer
  winDevHost->lsaFlatBase = (char*)devr->lsaFlatBase + win->bigOffset;
  winDevHost->mcOffset4K = win->bigOffset >> 12;
  winDevHost->stride4G = devr->bigSize >> 32;
  winDevHost->lsaRank = devr->lsaSelf;
  winDevHost->worldRank = comm->rank;
  winDevHost->winHost = (void*)win;
  winDevHost->ginOffset4K = memOffset >> 12;

  // Copy GIN window handles if GIN is enabled
  for (int i=0; i < NCCL_GIN_MAX_CONTEXTS; i++) {
    winDevHost->ginWins[i] = mem->ginDevWins[i];
  }

  // Step 4: Copy window metadata to device
  CUDACHECK(cudaMemcpyAsync(winDev, winDevHost, sizeof(struct ncclWindow_vidmem),
            cudaMemcpyHostToDevice, stream));

  // Step 5: Add window to device-side window table
  // This allows ncclFindWindow to locate windows from device code
  NCCLCHECK(symWindowTableInitOnce(comm, stream));
  struct ncclDevCommWindowTable* tableDev = devr->windowTable;

  while (true) {
    struct ncclDevCommWindowTable* tableHost;
    NCCLCHECK(ncclShadowPoolToHost(&devr->shadows, tableDev, &tableHost));

    // Find empty slot in table (32 entries per table)
    int i = 0;
    while (i < 32 && tableHost->entries[i].window != nullptr) i += 1;

    if (i < 32) {
      // Found empty slot - populate it
      tableHost->entries[i].base = userAddr;
      tableHost->entries[i].size = userSize;
      tableHost->entries[i].window = winDev;
      CUDACHECK(cudaMemcpyAsync(&tableDev->entries[i], &tableHost->entries[i],
                sizeof(tableHost->entries[i]), cudaMemcpyHostToDevice, stream));
      break;
    }

    // Table full - allocate next table in chain
    if (tableHost->next == nullptr) {
      NCCLCHECK(ncclShadowPoolAlloc<ncclDevCommWindowTable>(
                &devr->shadows, &tableHost->next, nullptr, stream));
      CUDACHECK(cudaMemcpyAsync(&tableDev->next, &tableHost->next,
                sizeof(tableHost->next), cudaMemcpyHostToDevice, stream));
    }
    tableDev = tableHost->next;
  }

  // Step 6: Add to sorted window list for fast lookup
  int i = listFindSortedLub(&ncclDevrWindowSorted::userAddr, devr->winSorted,
          devr->winSortedCount, userAddr);
  struct ncclDevrWindowSorted winSort;
  winSort.userAddr = userAddr;
  winSort.size = userSize;
  winSort.win = win;
  listInsert(&devr->winSorted, &devr->winSortedCapacity, &devr->winSortedCount,
             i, winSort);

  if (outWinDev) *outWinDev = winDev;
  if (outWin) *outWin = win;
  return ncclSuccess;
}
```

**Key Data Structure: ncclWindow_vidmem**

**File:** [src/include/nccl_device/impl/core__types.h:13-22](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/core__types.h#L13)

```cpp
struct ncclWindow_vidmem {
  void* winHost;              // Pointer back to host-side ncclDevrWindow
  char* lsaFlatBase;          // Pointer to rank 0's memory in flat VA space
  int lsaRank;                // This rank's LSA rank
  int worldRank;              // This rank's world rank
  uint32_t stride4G;          // Stride between ranks in units of 4GB
  uint32_t mcOffset4K;        // Offset for multicast in units of 4KB
  uint32_t ginOffset4K;       // Offset for GIN in units of 4KB
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONTEXTS]; // GIN handles
};
```

This structure is passed to device code and used by `ncclGetLsaPointer` to calculate peer memory addresses.

---

### 5.4 Device Communicator Creation: ncclDevCommCreate

The device communicator is the key Device API structure that gets passed to kernels.

#### User Code

**File:** [main.cu:169-184](/home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/01_allreduce/main.cu#L169)

```cpp
// Create stream for kernel execution
cudaStream_t stream;
CUDACHECK(cudaStreamCreate(&stream));

// Create device communicator - this is the key Device API component
// Requirements specify resources to allocate (e.g., one barrier per CTA)
ncclDevComm devComm;
ncclDevCommRequirements reqs;
memset(&reqs, 0, sizeof(reqs));

// Critical: Must allocate one LSA barrier per CTA
// This matches our kernel launch config: NCCL_DEVICE_CTA_COUNT = 16
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;

NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
printf("  Rank %d created device communicator with %d LSA barriers\n",
       my_rank, NCCL_DEVICE_CTA_COUNT);
```

**Why 16 barriers?**
- The kernel launches with 16 CTAs: `NCCL_DEVICE_CTA_COUNT = 16`
- Each CTA needs its own barrier for cross-GPU synchronization
- Barrier index in kernel matches `blockIdx.x`

#### Implementation: ncclDevCommCreate Entry Point

**File:** [src/dev_runtime.cc:966-1004](/home/jeromeku/torchcomms/thirdparty/nccl/src/dev_runtime.cc#L966)

```cpp
NCCL_API(ncclResult_t, ncclDevCommCreate,
         ncclComm_t comm, ncclDevCommRequirements_t const* reqs,
         ncclDevComm_t* outDevComm);

ncclResult_t ncclDevCommCreate(
    ncclComm_t comm, struct ncclDevCommRequirements const* reqs,
    struct ncclDevComm* outDevComm
  ) {
  ncclResult_t ret = ncclSuccess;
  int saveDev;
  struct ncclDevrCommCreateTask* task = nullptr;

  CUDACHECK(cudaGetDevice(&saveDev));
  NCCLCHECK(ncclGroupStartInternal());

  // Verify communicator supports symmetric memory
  if (!comm->symmetricSupport) {
    WARN("Communicator does not support symmetric memory!");
    ret = ncclInvalidUsage;
    goto fail;
  }

  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  // Initialize device runtime state
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  // Create task for deferred execution
  NCCLCHECKGOTO(ncclCalloc(&task, 1), ret, fail);

  // Deep copy requirements (will be accessed by background threads)
  NCCLCHECKGOTO(deepCopyDevCommRequirements(reqs, &task->reqs), ret, fail);
  task->outDevComm = outDevComm;

  // Enqueue task and join group
  ncclIntruQueueEnqueue(&comm->devrState.commCreateTaskQueue, task);
  ncclGroupCommJoin(comm, ncclGroupTaskTypeSymRegister);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  cudaSetDevice(saveDev);
  return ret;
fail:
  free(task);
  goto exit;
}
```

#### ncclDevrCommCreateInternal - Core Implementation

**File:** [src/dev_runtime.cc:709-886](/home/jeromeku/torchcomms/thirdparty/nccl/src/dev_runtime.cc#L709)

```cpp
ncclResult_t ncclDevrCommCreateInternal(
    struct ncclComm* comm,
    struct ncclDevCommRequirements const* reqs,
    struct ncclDevComm* outDevComm
  ) {
  ncclResult_t ret = ncclSuccess;
  struct ncclDevrState* devr = &comm->devrState;
  struct ncclTeam world = ncclTeamWorld(comm);
  struct ncclTeam lsa = ncclTeamInnerFactor(world, devr->lsaSize);
  bool ginActivated = false;
  struct ncclDevrTeam* tmLsa;
  size_t bufSizeTotal;
  int nGinContexts = 0;
  int ginSignalTotal = 0, ginCounterTotal = 0;
  struct ncclDevResourceRequirements* resReqsHead;
  struct ncclDevResourceRequirements lsaBarReq;
  cudaStream_t stream = nullptr;
  struct ncclDevResourceRequirements railGinBarrierReq;
  CUmemGenericAllocationHandle memHandle = 0x0;
  struct ncclDevrMemory* mem = nullptr;
  struct ncclDevrWindow* win = nullptr;
  struct ncclWindow_vidmem* winHost = nullptr;
  size_t ginSignalShadowsOffset = 0;

  // Step 1: Activate GIN if needed
  if (comm->nNodes > 1 || reqs->ginForceEnable ||
      reqs->ginCounterCount != 0 || reqs->ginSignalCount != 0) {
    ginActivated = !devr->ginEnabled;
    devr->ginEnabled = true;
  }

  if (ginActivated) {
    NCCLCHECKGOTO(ncclGinConnectOnce(comm), ret, fail);
    // Register all preexisting memories with GIN
    for (struct ncclDevrMemory* mem = devr->memHead; mem != nullptr;
         mem = mem->next) {
      NCCLCHECKGOTO(symMemoryRegisterGin(comm, mem), ret, fail);
    }
  }
  if (devr->ginEnabled) nGinContexts = comm->sharedRes->ginState.ginCommCount;

  // Step 2: Initialize device communicator base fields
  memset(outDevComm, 0, sizeof(*outDevComm));
  outDevComm->rank = comm->rank;
  outDevComm->nRanks = comm->nRanks;
  outDevComm->nRanks_rcp32 = idivRcp32(comm->nRanks);
  outDevComm->lsaRank = devr->lsaSelf;
  outDevComm->lsaSize = devr->lsaSize;
  outDevComm->lsaSize_rcp32 = idivRcp32(devr->lsaSize);

  // Step 3: Obtain LSA team (creates multicast memory if requested)
  NCCLCHECKGOTO(symTeamObtain(comm, lsa, /*multicast=*/reqs->lsaMultimem, &tmLsa),
                ret, fail);
  outDevComm->lsaMultimem.mcBasePtr = tmLsa->mcBasePtr;

  // Step 4: Process team requirements (for multimem)
  struct ncclTeamRequirements* tr = reqs->teamRequirementsList;
  while (tr != nullptr) {
    if (tr->multimem) {
      struct ncclDevrTeam* tm;
      NCCLCHECKGOTO(symTeamObtain(comm, tr->team, tr->multimem, &tm), ret, fail);
      if (tr->outMultimemHandle != nullptr)
        tr->outMultimemHandle->mcBasePtr = tm->mcBasePtr;
    }
    tr = tr->next;
  }

  // Step 5: Build resource requirements list
  resReqsHead = reqs->resourceRequirementsList;

  // Add LSA barrier requirement
  ncclLsaBarrierCreateRequirement(lsa,
    std::max(reqs->barrierCount, reqs->lsaBarrierCount),
    &outDevComm->lsaBarrier, &lsaBarReq);
  lsaBarReq.next = resReqsHead;
  resReqsHead = &lsaBarReq;

  // Add GIN barrier requirement
  ncclGinBarrierCreateRequirement(comm, ncclTeamRail(comm),
    std::max(reqs->barrierCount, reqs->railGinBarrierCount),
    &outDevComm->railGinBarrier, &railGinBarrierReq);
  railGinBarrierReq.next = resReqsHead;
  resReqsHead = &railGinBarrierReq;

  // Step 6: Calculate total buffer size needed
  struct ncclDevResourceRequirements* rr = resReqsHead;
  bufSizeTotal = 0;
  ginSignalTotal = reqs->ginSignalCount;
  ginCounterTotal = reqs->ginCounterCount;

  while (rr != nullptr) {
    // Align to 128 bytes (cache line) or required alignment
    bufSizeTotal = alignUp(bufSizeTotal, std::max<size_t>(128, rr->bufferAlign));

    // Assign buffer handle (offset / 128)
    if (rr->outBufferHandle != nullptr)
      *rr->outBufferHandle = bufSizeTotal / 128;

    // Assign GIN signal/counter ranges
    if (rr->outGinSignalStart != nullptr)
      *rr->outGinSignalStart = ginSignalTotal;
    if (rr->outGinCounterStart != nullptr)
      *rr->outGinCounterStart = ginCounterTotal;

    bufSizeTotal += rr->bufferSize;
    ginSignalTotal += rr->ginSignalCount;
    ginCounterTotal += rr->ginCounterCount;
    rr = rr->next;
  }

  bufSizeTotal = alignUp(bufSizeTotal, 128);
  ginSignalShadowsOffset = bufSizeTotal;

  // Add space for GIN signal shadows
  bufSizeTotal += nGinContexts * ginSignalTotal * sizeof(uint64_t);
  bufSizeTotal = alignUp(bufSizeTotal, devr->granularity);

  // Step 7: Create CUDA stream for async operations
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                ret, fail);

  // Step 8: Update GIN handles in existing windows if GIN was just activated
  if (ginActivated) {
    for (int i=0; i < devr->winSortedCount; i++) {
      struct ncclDevrWindow* win = devr->winSorted[i].win;
      struct ncclWindow_vidmem* winHost;
      NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, win->vidmem, &winHost),
                    ret, fail_stream);
      winHost->ginOffset4K = (win->bigOffset - win->memory->bigOffset) >> 12;
      for (int i=0; i < NCCL_GIN_MAX_CONTEXTS; i++) {
        winHost->ginWins[i] = win->memory->ginDevWins[i];
      }
      CUDACHECKGOTO(cudaMemcpyAsync(win->vidmem, winHost,
                    sizeof(struct ncclWindow_vidmem),
                    cudaMemcpyHostToDevice, stream), ret, fail_stream);
    }
  }

  // Step 9: Ensure window table exists
  NCCLCHECKGOTO(symWindowTableInitOnce(comm, stream), ret, fail_stream);
  outDevComm->windowTable = devr->windowTable;

  // Step 10: Allocate and map resource buffer if needed
  if (bufSizeTotal == 0) {
    outDevComm->resourceWindow = nullptr;
    outDevComm->resourceWindow_inlined = {};
  } else {
    CUmemAllocationProp memProp = {};
    memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memProp.requestedHandleTypes = ncclCuMemHandleType;

    // Enable RDMA if GIN might be used
    memProp.allocFlags.gpuDirectRDMACapable =
      comm->sharedRes->ginState.ncclGin != nullptr ? 1 : 0;
    memProp.location.id = comm->cudaDev;

    // Create memory handle
    CUCHECKGOTO(cuMemCreate(&memHandle, bufSizeTotal, &memProp, 0),
                ret, fail_stream);

    // Register as symmetric memory
    NCCLCHECKGOTO(symMemoryObtain(comm, memHandle, NULL, bufSizeTotal, &mem),
                  ret, fail_stream_mem);
    memHandle = 0x0; // Reference given to symMemoryObtain

    // Create window for resource buffer
    NCCLCHECKGOTO(symWindowCreate(
      comm, mem, /*memOffset=*/0, nullptr, bufSizeTotal, /*winFlags=*/0,
      /*localReg=*/nullptr, &outDevComm->resourceWindow, &win,
      stream), ret, fail_stream_mem);
    mem = nullptr; // Reference given to symWindowCreate

    // Copy window metadata to device comm
    NCCLCHECKGOTO(ncclShadowPoolToHost(&devr->shadows, win->vidmem, &winHost),
                  ret, fail_stream_mem_win);
    outDevComm->resourceWindow_inlined = *winHost;

    // Calculate GIN signal shadows pointer
    outDevComm->ginSignalShadows = (uint64_t*)add4G(
      (char*)winHost->lsaFlatBase + ginSignalShadowsOffset,
      winHost->lsaRank * winHost->stride4G
    );

    // Zero-initialize resource buffer
    CUDACHECKGOTO(cudaMemsetAsync(win->userPtr, 0, bufSizeTotal, stream),
                  ret, fail_stream_mem_win);
  }

  // Step 11: Allocate GIN signals and counters
  if (devr->ginEnabled) {
    outDevComm->ginContextCount = nGinContexts;
    outDevComm->ginSignalCount = ginSignalTotal;
    outDevComm->ginCounterCount = ginCounterTotal;

    NCCLCHECKGOTO(ncclGinAllocSignalsCounters(comm,
      ginSignalTotal, &outDevComm->ginSignalBase,
      ginCounterTotal, &outDevComm->ginCounterBase
    ), ret, fail_stream_mem_win);

    // Copy GIN handles
    for (int ctx=0; ctx < nGinContexts; ctx++) {
      outDevComm->ginTypes[ctx] =
        (int)comm->sharedRes->ginState.ginDevHandles[ctx]->netDeviceType;
      outDevComm->ginHandles[ctx] =
        comm->sharedRes->ginState.ginDevHandles[ctx]->handle;
    }
  }

  // Step 12: Synchronize stream and barrier across ranks
  CUDACHECKGOTO(cudaStreamSynchronize(stream), ret, fail_stream_mem_win_signals);
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->rank, comm->nRanks, 0xbeef),
                ret, fail_stream_mem_win_signals);

  CUDACHECKGOTO(cudaStreamDestroy(stream), ret, fail_stream_mem_win_signals);
  return ret;

// Error handling omitted for brevity
}
```

#### LSA Barrier Requirement Calculation

**File:** [src/nccl_device/lsa_barrier.cc:10-21](/home/jeromeku/torchcomms/thirdparty/nccl/src/nccl_device/lsa_barrier.cc#L10)

```cpp
NCCL_API(ncclResult_t, ncclLsaBarrierCreateRequirement,
         ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle,
         ncclDevResourceRequirements_t* outReq);

ncclResult_t ncclLsaBarrierCreateRequirement(
    ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle,
    ncclDevResourceRequirements_t* outReq
  ) {
  memset(outReq, 0, sizeof(*outReq));
  outHandle->nBarriers = nBarriers;

  // Calculate buffer size for barrier state
  // Layout: [epoch state][inbox counters]
  //   - 3*nBarriers uint32_t for epoch state
  //   - nBarriers*team.nRanks uint32_t for peer inboxes
  outReq->bufferSize = (3*nBarriers + nBarriers*team.nRanks) * sizeof(uint32_t);
  outReq->bufferAlign = alignof(uint32_t);
  outReq->outBufferHandle = &outHandle->bufHandle;

  return ncclSuccess;
}
```

**Barrier Memory Layout (for 16 barriers, 8 LSA ranks):**

```
Resource Buffer Layout:
  [Epoch State: 3 * 16 * 4 bytes = 192 bytes]
    - Used for multicast and unicast epoch tracking

  [Peer Inboxes: 16 * 8 * 4 bytes = 512 bytes]
    - Each barrier has 8 inboxes (one per peer in LSA team)
    - Used for unicast barrier synchronization
```

---

## Device-Side Execution Traces

### 6.1 Kernel Launch

**File:** [main.cu:194-200](/home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/01_allreduce/main.cu#L194)

```cpp
// Launch device kernel to perform AllReduce
// Grid: 16 CTAs, each with 512 threads
// Parameters:
//   - send_win, recv_win: Symmetric window handles
//   - count: Number of elements (1M floats)
//   - devComm: Device communicator with barriers and LSA info
simpleAllReduceKernel<<<NCCL_DEVICE_CTA_COUNT, NCCL_DEVICE_THREADS_PER_CTA,
                        0, stream>>>(
  send_win, 0,      // Send window and offset
  recv_win, 0,      // Receive window and offset
  count, 0,         // Element count and root
  devComm           // Device communicator
);

// Wait for kernel completion
CUDACHECK(cudaStreamSynchronize(stream));
```

**Launch Configuration:**
- Grid: `16 CTAs` (matches `lsaBarrierCount`)
- Block: `512 threads per CTA`
- Total threads per rank: `16 * 512 = 8192 threads`

---

### 6.2 User Kernel: simpleAllReduceKernel - Line by Line

**File:** [main.cu:59-98](/home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/01_allreduce/main.cu#L59)

```cpp
__global__ void simpleAllReduceKernel(
    ncclWindow_t sendwin, size_t sendoffset,
    ncclWindow_t recvwin, size_t recvoffset,
    size_t count, int root, struct ncclDevComm devComm
  ) {
```

**Kernel Parameters:**
- `sendwin`, `recvwin`: Window handles (pointers to `ncclWindow_vidmem`)
- `sendoffset`, `recvoffset`: Byte offsets within windows
- `count`: Number of elements to process
- `root`: Root rank (unused in AllReduce)
- `devComm`: Device communicator structure (passed by value)

#### Line 67-68: LSA Barrier Session Initialization

```cpp
  // LSA barriers enable coordination between GPU threads across different ranks
  // Barrier scope: CTA (all threads in this block participate)
  // Barrier index: blockIdx.x selects this CTA's dedicated barrier
  ncclLsaBarrierSession<ncclCoopCta> bar {
    ncclCoopCta(),              // Cooperation scope: entire CTA
    devComm,                    // Device communicator
    ncclTeamLsa(devComm),       // LSA team (local GPUs)
    devComm.lsaBarrier,         // Barrier handle
    blockIdx.x                  // Barrier index (0-15)
  };
```

**What this does:**
1. Creates a barrier session for this CTA
2. Each CTA gets its own barrier (index = `blockIdx.x`)
3. Barrier coordinates across all CTAs with same index on different GPUs
4. Uses `ncclCoopCta` cooperation scope (all threads in block)

**Constructor Implementation:** [src/include/nccl_device/impl/lsa_barrier__funcs.h:14-30](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/lsa_barrier__funcs.h#L14)

```cpp
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::ncclLsaBarrierSession(
    Coop coop, ncclDevComm const& comm, ncclTeam team,
    ncclLsaBarrierHandle handle, uint32_t index,
    bool multimem, ncclMultimemHandle mmHandle
  ):
  ncclLsaBarrierSession_internal<Coop>{
    coop, comm, team, handle, (int)index,
    multimem, mmHandle, /*epoch=*/0
  } {
  // Load current epoch from barrier state
  uint32_t* state = (uint32_t*)ncclGetResourceBufferLocalPointer(comm,
                                                                  handle.bufHandle);
  this->epoch = state[(this->multimem ? 0 : 1)*this->handle.nBarriers + this->index];
}
```

**Barrier Session Internal Structure:** [src/include/nccl_device/impl/lsa_barrier__types.h:18-43](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/lsa_barrier__types.h#L18)

```cpp
template<typename Coop>
struct ncclLsaBarrierSession_internal {
  Coop coop;                          // Cooperation scope (CTA, warp, etc.)
  ncclDevComm const& comm;            // Reference to device communicator
  ncclTeam team;                      // LSA team info
  ncclLsaBarrierHandle handle;        // Barrier handle
  int index;                          // Barrier index
  bool multimem;                      // Using multicast memory?
  ncclMultimemHandle mmHandle;        // Multicast handle
  uint32_t epoch;                     // Current epoch counter

  // Get multicast inbox pointer
  NCCL_DEVICE_INLINE uint32_t* mcInbox(bool multimem) {
    uint32_t* state;
    if (multimem) {
      state = (uint32_t*)ncclGetResourceBufferMultimemPointer(comm,
                         handle.bufHandle, mmHandle);
    } else {
      state = (uint32_t*)ncclGetResourceBufferLocalPointer(comm,
                         handle.bufHandle);
    }
    return state + 2*handle.nBarriers + index;
  }

  // Get unicast inbox pointer
  NCCL_DEVICE_INLINE uint32_t* ucInbox(int owner, int peer) {
    uint32_t* state = (uint32_t*)ncclGetResourceBufferPeerPointer(comm,
                       handle.bufHandle, team, owner);
    return state + 3*handle.nBarriers + index*team.nRanks + peer;
  }
};
```

#### Line 69: Entry Barrier Synchronization

```cpp
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
```

**What this does:**
1. All threads in CTA synchronize locally via `__syncthreads()`
2. Thread 0 signals all peers that this CTA has arrived
3. All threads wait for all peers' CTAs to arrive
4. Uses `relaxed` memory ordering (no fence needed yet)

**Implementation:** [src/include/nccl_device/impl/lsa_barrier__funcs.h:118-124](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/lsa_barrier__funcs.h#L118)

```cpp
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::sync(
    Coop coop, cuda::memory_order order
  ) {
  this->arrive(coop, order);  // Signal peers
  this->wait(coop, order);    // Wait for peers
}
```

**Arrive Implementation:** [src/include/nccl_device/impl/lsa_barrier__funcs.h:62-84](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/lsa_barrier__funcs.h#L62)

```cpp
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::arrive(
    Coop, cuda::memory_order order
  ) {
  this->coop.sync();  // Local CTA barrier first

  if (this->multimem) {
    // Multicast path (not used in this example)
    #if __CUDA_ARCH__ >= 900
    if (this->coop.thread_rank() == 0) {
      uint32_t* inbox = this->mcInbox(/*multimem=*/true);
      if (nccl::utility::releaseOrderOf(order) != cuda::memory_order_relaxed) {
        asm volatile("multimem.red.release.sys.add.u32 [%0],1;" :: "l"(inbox));
      } else {
        asm volatile("multimem.red.relaxed.sys.add.u32 [%0],1;" :: "l"(inbox));
      }
    }
    #endif
  } else {
    // Unicast path: write to each peer's inbox
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < this->team.nRanks-1;
         i += this->coop.size()) {
      // Calculate peer rank (skip self)
      int peer = i + (this->team.rank <= i ? 1 : 0);

      // Atomic store to peer's inbox
      cuda::atomic_ref<uint32_t> inbox(*this->ucInbox(peer, this->team.rank));
      inbox.store(this->epoch+1, nccl::utility::releaseOrderOf(order));
    }
  }
}
```

**Wait Implementation:** [src/include/nccl_device/impl/lsa_barrier__funcs.h:86-116](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/lsa_barrier__funcs.h#L86)

```cpp
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::wait(
    Coop, cuda::memory_order order
  ) {
  if (this->multimem) {
    // Multicast path (not used in this example)
    #if __CUDA_ARCH__ >= 900
    if (this->coop.thread_rank() == 0) {
      cuda::atomic_ref<uint32_t> inbox(*this->mcInbox(/*multimem=*/false));
      #pragma unroll 1
      while (true) {
        uint32_t got = inbox.load(nccl::utility::acquireOrderOf(order));
        if (got - (this->epoch + this->team.nRanks) <= uint32_t(-1)>>1) break;
      }
      this->epoch += this->team.nRanks;
    }
    #endif
  } else {
    // Unicast path: poll each peer's write to our inbox
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < this->team.nRanks-1;
         i += this->coop.size()) {
      int peer = i + (this->team.rank <= i ? 1 : 0);

      // Atomic load from our inbox (peer writes to it)
      cuda::atomic_ref<uint32_t> inbox(*this->ucInbox(this->team.rank, peer));
      #pragma unroll 1
      while (true) {
        uint32_t got = inbox.load(nccl::utility::acquireOrderOf(order));
        // Check if peer has incremented epoch
        if (got - (this->epoch + 1) <= uint32_t(-1)>>1) break;
      }
    }
    this->epoch += 1;
  }
  this->coop.sync();  // Local CTA barrier after wait
}
```

**Barrier Synchronization Pattern:**

For 4 GPUs (LSA ranks 0-3), CTA with index 5:

```
GPU 0 CTA 5                    GPU 1 CTA 5                    GPU 2 CTA 5
+-----------+                  +-----------+                  +-----------+
| arrive(): |                  | arrive(): |                  | arrive(): |
| Write 1→1 |----------------->| Inbox[0]  |                  |           |
| Write 1→2 |---------------------------------------->        | Inbox[0]  |
| Write 1→3 |---------------------------------------------------------->| Inbox[0]  |
|           |                  | Write 1→0 |<-----------------+           |
|           |                  | Write 1→2 |----------------->| Inbox[1]  |
|           |                  | Write 1→3 |-------------------------------->| Inbox[1]  |
|           |<-----------------| Write 1→0 |                  |           |
| Inbox[1]  |                  |           |                  | Write 1→0 |
| Inbox[2]  |<----------------------------------------        | Write 1→1 |
| Inbox[3]  |<--------------------------------------------------------+Write 1→3 |
|           |                  |           |                  |           |
| wait():   |                  | wait():   |                  | wait():   |
| Poll Inbox[1,2,3]            | Poll Inbox[0,2,3]            | Poll Inbox[0,1,3]
| See all == 1                 | See all == 1                 | See all == 1
| epoch++   |                  | epoch++   |                  | epoch++   |
+-----------+                  +-----------+                  +-----------+
```

#### Line 71: Extract Rank Information

```cpp
  const int rank = devComm.rank, nRanks = devComm.nRanks;
```

**Device Communicator Access:**
- `devComm` is passed by value to kernel
- `rank`: This rank's world rank (0-based)
- `nRanks`: Total number of ranks in communicator

#### Line 73-77: Calculate Global Thread ID

```cpp
  // We are going to spread the workload across all GPU ranks.
  // So calculate the global thread ID across all ranks.
  // This maps global threads to data elements in the data to be reduced
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;
```

**Thread Mapping:**

For 4 ranks, 16 CTAs, 512 threads per CTA:
- Total threads per rank: `16 * 512 = 8,192`
- Total threads globally: `8,192 * 4 = 32,768`

**Example thread IDs:**

| Rank | Block | Thread | globalTid Calculation | globalTid Value |
|------|-------|--------|----------------------|-----------------|
| 0 | 0 | 0 | 0 + 512 * (0 + 0*4) = 0 | 0 |
| 0 | 0 | 511 | 511 + 512 * (0 + 0*4) = 511 | 511 |
| 0 | 1 | 0 | 0 + 512 * (0 + 1*4) = 2048 | 2048 |
| 1 | 0 | 0 | 0 + 512 * (1 + 0*4) = 512 | 512 |
| 1 | 1 | 0 | 0 + 512 * (1 + 1*4) = 2560 | 2560 |
| 2 | 0 | 0 | 0 + 512 * (2 + 0*4) = 1024 | 1024 |
| 3 | 0 | 0 | 0 + 512 * (3 + 0*4) = 1536 | 1536 |

**Pattern:** Threads are interleaved by rank within each block's global position.

#### Line 80-93: Grid-Stride Loop with AllReduce Logic

```cpp
  // Grid stride loop over all elements with the globalThreads
  for (size_t offset = globalTid; offset < count; offset += globalNthreads) {
    float v = 0;

    // Phase 1: Access remote (and local) memory and reduce locally
    for (int peer=0; peer<nRanks; peer++) {
      // Access peer memory directly using LSA (Load/Store Accessible) pointers
      float* sendPtr = (float*)ncclGetLsaPointer(sendwin, sendoffset, peer);
      v += sendPtr[offset];
    }

    // Phase 2: Write the result back to remote and local memory
    for (int peer=0; peer<nRanks; peer++) {
      float* recvPtr = (float*)ncclGetLsaPointer(recvwin, recvoffset, peer);
      recvPtr[offset] = v;
    }
  }
```

**What happens:**

1. **Grid-Stride Loop:**
   - Each thread processes elements at stride `globalNthreads`
   - Example: Thread 0 processes elements `[0, 32768, 65536, ...]`
   - Ensures good load balancing even if `count` is not a multiple of thread count

2. **Phase 1 - Reduce:**
   - Each thread reads element `offset` from all peers
   - Accumulates into local variable `v`
   - Uses direct peer memory access (no NCCL kernel needed)

3. **Phase 2 - Broadcast:**
   - Each thread writes accumulated result to all peers
   - Result is replicated across all ranks (AllReduce)

**Memory Access Pattern for Element 0 (4 ranks):**

```
Rank 0 Thread 0:                Rank 1 Thread 512:
  Read sendPtr[0] from:           Read sendPtr[0] from:
    Rank 0 (local)                  Rank 0 (remote via RDMA)
    Rank 1 (remote via RDMA)        Rank 1 (local)
    Rank 2 (remote via RDMA)        Rank 2 (remote via RDMA)
    Rank 3 (remote via RDMA)        Rank 3 (remote via RDMA)
  Compute: v = sum of 4 values    Compute: v = sum of 4 values

  Write v to recvPtr[0] on:       Write v to recvPtr[0] on:
    Rank 0 (local)                  Rank 0 (remote via RDMA)
    Rank 1 (remote via RDMA)        Rank 1 (local)
    Rank 2 (remote via RDMA)        Rank 2 (remote via RDMA)
    Rank 3 (remote via RDMA)        Rank 3 (remote via RDMA)
```

---

### 6.3 ncclGetLsaPointer - Device-Side Pointer Calculation

**Call Site:** [main.cu:85](/home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/01_allreduce/main.cu#L85)

```cpp
float* sendPtr = (float*)ncclGetLsaPointer(sendwin, sendoffset, peer);
```

**Implementation:** [src/include/nccl_device/impl/core__funcs.h:112-119](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/core__funcs.h#L112)

```cpp
#if __CUDACC__
NCCL_DEVICE_INLINE void* ncclGetLsaPointer(
    ncclWindow_t w, size_t offset, int peer
  ) {
  // Load window metadata from constant memory
  char* base = nccl::utility::loadConst(&w->lsaFlatBase);
  uint32_t stride4G = nccl::utility::loadConst(&w->stride4G);

  // peer is the LSA rank to access
  int i = peer;

  // Calculate address:
  //   base: Points to rank 0's memory
  //   add4G: Add peer*stride4G in units of 4GB
  //   + offset: Add byte offset within window
  return (void*)(nccl::utility::add4G(base, i*stride4G) + offset);
}
#endif
```

**Address Calculation Breakdown:**

Given:
- `lsaFlatBase = 0x100000000000` (rank 0's base in flat VA)
- `stride4G = 32` (128GB / 4GB = 32)
- `peer = 2` (want rank 2's memory)
- `offset = 1024` (byte offset in window)

Calculation:
```
base = 0x100000000000
stride_bytes = 32 * 4GB = 128GB = 0x2000000000
peer_base = base + (2 * 0x2000000000) = 0x100040000000
final_ptr = 0x100040000000 + 1024 = 0x100040000400
```

**Symmetric Property:**

All ranks compute the **same address** for a given `(peer, offset)`:
- Rank 0 accessing peer 2's memory → `0x100040000400`
- Rank 1 accessing peer 2's memory → `0x100040000400`
- Rank 2 accessing its own memory (peer=2) → `0x100040000400`

This is the power of symmetric addressing!

---

### 6.4 Exit Barrier with Release Semantics

**File:** [main.cu:94-97](/home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/01_allreduce/main.cu#L94)

```cpp
  // Release barrier ensures that we received data from everyone before
  // we unblock the stream and allow the next kernel(s) to process the data.
  // Critical for correctness in device-side collective operations
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}
```

**Why `memory_order_release`?**

1. **Entry barrier** (`relaxed`): No memory ordering needed, just synchronization
2. **Computation**: Direct loads/stores to peer memory
3. **Exit barrier** (`release`): Ensures all stores are visible before unblocking

**Memory Ordering Semantics:**

```cpp
// Without release barrier:
GPU 0 Thread:                   GPU 1 Thread:
  recvPtr[0] = v;                 (may not see GPU 0's write yet)
  bar.sync(relaxed);              bar.sync(relaxed);
  return;                         float x = recvPtr[0]; // RACE!

// With release barrier:
GPU 0 Thread:                   GPU 1 Thread:
  recvPtr[0] = v;                 (waiting...)
  bar.sync(release);              bar.sync(release);
    -> fence ensures write        -> sees GPU 0's write
       is visible                 float x = recvPtr[0]; // SAFE!
  return;                         return;
```

**Barrier Destructor:** [src/include/nccl_device/impl/lsa_barrier__funcs.h:44-58](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/lsa_barrier__funcs.h#L44)

```cpp
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::~ncclLsaBarrierSession() {
  // Save epoch back to barrier state
  uint32_t* state = (uint32_t*)ncclGetResourceBufferLocalPointer(
                      this->comm, this->handle.bufHandle);

  if (this->coop.thread_rank() == 0) {
    // Only thread 0 updates epoch
    state[(this->multimem ? 0 : 1)*this->handle.nBarriers + this->index]
      = this->epoch;
  }

  // Final CTA barrier
  this->coop.sync();
}
```

**Epoch Persistence:**
- Epoch is saved so next kernel launch can continue from where it left off
- Prevents ABA problems in barrier synchronization
- Each barrier invocation increments epoch

---

## LSA Barrier Deep Dive

### 7.1 What is an LSA Barrier?

**LSA (Load/Store Accessible) Barrier** is a cross-GPU synchronization primitive that enables device code to coordinate execution across multiple GPUs without CPU intervention.

**Key Properties:**

1. **Cross-GPU Scope**: Synchronizes threads across different GPUs
2. **Device-Initiated**: No host CPU involvement
3. **Epoch-Based**: Uses incrementing epoch counters to avoid ABA problems
4. **Memory-Backed**: State stored in symmetric memory accessible by all GPUs
5. **Peer-to-Peer**: Uses direct GPU-to-GPU memory writes (RDMA)

**How it differs from other synchronization:**

| Primitive | Scope | Initiator | Mechanism |
|-----------|-------|-----------|-----------|
| `__syncthreads()` | Single CTA | Device | Barrier instruction |
| Cooperative Groups | Single GPU | Device | Barrier instruction |
| CUDA Events | Stream | Host | PCIe signaling |
| LSA Barrier | Cross-GPU | Device | Peer memory atomics |

---

### 7.2 LSA Barrier Structure and Handles

**ncclLsaBarrierHandle:** [src/include/nccl_device/impl/lsa_barrier__types.h:12-15](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/lsa_barrier__types.h#L12)

```cpp
struct ncclLsaBarrierHandle {
  ncclDevResourceHandle_t bufHandle;  // Handle to resource buffer
  int nBarriers;                      // Number of barriers allocated
};
```

**ncclLsaBarrierSession:** The templated wrapper for barrier usage

```cpp
template<typename Coop>
struct ncclLsaBarrierSession {
  // Constructor: Initialize barrier for this CTA
  ncclLsaBarrierSession(Coop, ncclDevComm const&, ncclTeam,
                        ncclLsaBarrierHandle, uint32_t index,
                        bool multimem=false, ncclMultimemHandle={});

  // Destructor: Save epoch state
  ~ncclLsaBarrierSession();

  // Barrier operations
  void arrive(Coop, cuda::memory_order);  // Signal arrival
  void wait(Coop, cuda::memory_order);    // Wait for others
  void sync(Coop, cuda::memory_order);    // arrive + wait
};
```

---

### 7.3 Barrier Memory Layout

For `nBarriers = 16`, `team.nRanks = 8`:

```
Resource Buffer:
  [0:192 bytes]   - Epoch State
    [0:64 bytes]    - Multicast epoch state (3 * 16 * 4 bytes)
    [64:128 bytes]  - Unicast epoch state (3 * 16 * 4 bytes)
    [128:192 bytes] - Reserved

  [192:704 bytes] - Peer Inboxes
    For each barrier i (0-15):
      For each rank r (0-7):
        inbox[i][r] at offset: 192 + (i * 8 + r) * 4
```

**Inbox Layout for Barrier 5:**

```
Rank 0's View:
  inbox[5][0] -> Rank 0's inbox (written by rank 1,2,3,4,5,6,7)
  inbox[5][1] -> Rank 1's inbox (at rank 1's memory)
  inbox[5][2] -> Rank 2's inbox (at rank 2's memory)
  ...

Rank 1's View:
  inbox[5][0] -> Rank 0's inbox (at rank 0's memory)
  inbox[5][1] -> Rank 1's inbox (written by rank 0,2,3,4,5,6,7)
  inbox[5][2] -> Rank 2's inbox (at rank 2's memory)
  ...
```

---

### 7.4 Barrier Synchronization Algorithm

**Unicast Algorithm (used in example):**

```
Phase 1: Arrive
  For each peer p (excluding self):
    Atomic store (epoch+1) to peer's inbox[barrier_id][my_rank]

Phase 2: Wait
  For each peer p (excluding self):
    While inbox[barrier_id][p] < (epoch+1):
      Spin (atomic load)
  epoch++
```

**Multicast Algorithm (for NVLS-capable hardware):**

```
Phase 1: Arrive
  Atomic increment multicast inbox (visible to all ranks)

Phase 2: Wait
  While multicast inbox < (epoch + team.nRanks):
    Spin
  epoch += team.nRanks
```

---

### 7.5 Memory Ordering in Barriers

**CUDA C++ Memory Orders:**

| Memory Order | Meaning | Used For |
|--------------|---------|----------|
| `relaxed` | No fence, just atomicity | Synchronization only |
| `acquire` | Fence before load | See peer writes |
| `release` | Fence after store | Make writes visible |
| `acq_rel` | Both acquire and release | Full fence |

**Barrier Usage Patterns:**

```cpp
// Pattern 1: Synchronization only (no data dependency)
bar.sync(coop, cuda::memory_order_relaxed);

// Pattern 2: Release writes before barrier
store_to_peer_memory();
bar.sync(coop, cuda::memory_order_release);  // Fence ensures stores visible

// Pattern 3: Acquire peer writes after barrier
bar.sync(coop, cuda::memory_order_acquire);
load_from_peer_memory();  // Guaranteed to see peer writes

// Pattern 4: Full fence (release + acquire)
store_to_peer_memory();
bar.sync(coop, cuda::memory_order_acq_rel);
load_from_peer_memory();
```

**In the example:**
- Entry barrier: `relaxed` (no data dependency yet)
- Exit barrier: `release` (ensure all stores are visible before returning)

---

### 7.6 Why One Barrier Per CTA?

**Problem:** If all CTAs shared one barrier, they would all block each other:

```
Bad: 1 barrier for 16 CTAs
  CTA 0 on Rank 0 arrives -> waits for all 16*4=64 CTAs
  CTA 1 on Rank 0 arrives -> waits for all 64 CTAs
  ...
  -> Massive serialization!
```

**Solution:** Each CTA gets its own barrier:

```
Good: 16 barriers for 16 CTAs
  CTA 0 uses barrier 0 -> waits for 4 CTAs (one per rank)
  CTA 1 uses barrier 1 -> waits for 4 CTAs (one per rank)
  ...
  -> Parallel synchronization!
```

**Barrier Index Selection:**
```cpp
ncclLsaBarrierSession<ncclCoopCta> bar {
  ...,
  blockIdx.x  // Each CTA uses its block ID as barrier index
};
```

**Synchronization Groups:**

```
Barrier 0:  CTA 0 @ Rank 0, CTA 0 @ Rank 1, CTA 0 @ Rank 2, CTA 0 @ Rank 3
Barrier 1:  CTA 1 @ Rank 0, CTA 1 @ Rank 1, CTA 1 @ Rank 2, CTA 1 @ Rank 3
...
Barrier 15: CTA 15 @ Rank 0, CTA 15 @ Rank 1, CTA 15 @ Rank 2, CTA 15 @ Rank 3
```

Each group synchronizes independently!

---

## Device Communicator Structure

### 8.1 ncclDevComm Structure

**File:** [src/include/nccl_device/impl/comm__types.h:26-49](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/comm__types.h#L26)

```cpp
struct ncclDevComm {
  // Basic rank info
  int rank, nRanks;
  uint32_t nRanks_rcp32;        // Reciprocal for fast division

  // LSA team info
  int lsaRank, lsaSize;
  uint32_t lsaSize_rcp32;       // Reciprocal for fast division

  // Window table for buffer lookup
  struct ncclDevCommWindowTable* windowTable;

  // Resource buffer (for barriers, etc.)
  ncclWindow_t resourceWindow;
  struct ncclWindow_vidmem resourceWindow_inlined;

  // Multicast memory
  ncclMultimemHandle_t lsaMultimem;

  // Barrier handles
  ncclLsaBarrierHandle_t lsaBarrier;
  ncclGinBarrierHandle_t railGinBarrier;

  // GIN (GPU Initiated Network) state
  uint8_t ginContextCount;
  uint8_t ginTypes[4];
  void* ginHandles[4];
  uint32_t ginSignalBase;
  int ginSignalCount;
  uint32_t ginCounterBase;
  int ginCounterCount;
  uint64_t* ginSignalShadows;
};
```

**Key Fields:**

1. **rank, nRanks**: Standard NCCL rank information
2. **lsaRank, lsaSize**: Position within LSA team (local GPUs)
3. **windowTable**: Device-side lookup table for registered windows
4. **resourceWindow**: Buffer for barrier state and internal resources
5. **lsaBarrier**: Handle to LSA barrier allocation
6. **ginHandles**: For inter-node communication (if enabled)

---

### 8.2 ncclDevCommRequirements Structure

**File:** [src/include/nccl_device/core.h:62-78](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/core.h#L62)

```cpp
struct ncclDevCommRequirements {
  // Linked lists of resource requirements
  ncclDevResourceRequirements_t* resourceRequirementsList;
  ncclTeamRequirements_t* teamRequirementsList;

  // Multicast settings
  bool lsaMultimem;             // Enable multicast on LSA team

  // Barrier counts
  int barrierCount;             // Generic barriers
  int lsaBarrierCount;          // LSA-specific barriers
  int railGinBarrierCount;      // GIN barriers for multi-node

  // Low-Latency All-to-All settings
  int lsaLLA2ABlockCount, lsaLLA2ASlotCount;

  // GIN settings
  bool ginForceEnable;          // Force GIN activation
  int ginContextCount;          // Number of GIN contexts (hint)
  int ginSignalCount;           // Number of GIN signals
  int ginCounterCount;          // Number of GIN counters
};
```

**Usage Pattern:**

```cpp
ncclDevCommRequirements reqs;
memset(&reqs, 0, sizeof(reqs));

// Request 16 LSA barriers (one per CTA)
reqs.lsaBarrierCount = 16;

// Optionally enable multicast
reqs.lsaMultimem = true;

ncclDevComm devComm;
ncclDevCommCreate(comm, &reqs, &devComm);
```

---

### 8.3 ncclTeam Structures

**ncclTeam:** [src/include/nccl_device/core.h:39-41](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/core.h#L39)

```cpp
struct ncclTeam {
  int nRanks;  // Number of ranks in team
  int rank;    // This rank's position in team
  int stride;  // Stride between consecutive team members in world
};
```

**Team Types:**

1. **World Team:**
   ```cpp
   ncclTeam world = ncclTeamWorld(devComm);
   // nRanks = comm->nRanks
   // rank = comm->rank
   // stride = 1
   ```

2. **LSA Team:**
   ```cpp
   ncclTeam lsa = ncclTeamLsa(devComm);
   // nRanks = devComm.lsaSize (e.g., 8 local GPUs)
   // rank = devComm.lsaRank (0-7)
   // stride = 1
   ```

3. **Rail Team:**
   ```cpp
   ncclTeam rail = ncclTeamRail(devComm);
   // nRanks = comm->nRanks / lsaSize (e.g., number of nodes)
   // rank = comm->rank / lsaSize
   // stride = lsaSize
   ```

**Team Tag Types:** [src/include/nccl_device/core.h:48-50](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/core.h#L48)

```cpp
struct ncclTeamTagWorld {};
struct ncclTeamTagLsa {};
struct ncclTeamTagRail {};
```

Used for type-safe team construction:
```cpp
ncclLsaBarrierSession<ncclCoopCta> bar {
  coop, devComm,
  ncclTeamTagLsa(),  // Type tag for LSA team
  index
};
```

---

### 8.4 Window Table Structure

**File:** [src/include/nccl_device/impl/comm__types.h:16-24](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/comm__types.h#L16)

```cpp
struct ncclDevCommWindowTable {
  struct Entry {
    uintptr_t base, size;       // User pointer range
    ncclWindow_t window;        // Associated window handle
  } entries[32];

  struct ncclDevCommWindowTable* next;  // Linked list for overflow
};
```

**Purpose:** Device-side lookup of windows from user pointers

**Usage in Device Code:**

```cpp
// Find window containing ptr
ncclWindow_t win = ncclFindWindow(coop, devComm, ptr);

// Now can use window to access peer memory
void* peerPtr = ncclGetLsaPointer(win, offset, peer);
```

---

## Comparison with Host API

### 9.1 Same Aspects

| Aspect | Details |
|--------|---------|
| **Communicator Init** | Both use `ncclCommInitRank` to create `ncclComm_t` |
| **Memory Allocation** | Both can use `ncclMemAlloc` for VMM support |
| **Window Registration** | Both can use `ncclCommWindowRegister` (Device API requires it) |
| **CUDA Streams** | Both use CUDA streams for async execution |
| **Cleanup** | Both use `ncclCommDestroy`, `ncclMemFree` |

---

### 9.2 Different Aspects

#### Communication Initiation

**Host API:**
```cpp
// CPU initiates communication
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
cudaStreamSynchronize(stream);
```

**Device API:**
```cpp
// GPU kernel performs communication directly
myKernel<<<grid, block, 0, stream>>>(sendwin, recvwin, count, devComm);
cudaStreamSynchronize(stream);
```

#### Synchronization

**Host API:**
- CUDA stream ordering
- CUDA events
- CPU-side barriers

**Device API:**
- LSA barriers for cross-GPU sync
- Memory ordering semantics (`acquire`/`release`)
- Epoch-based coordination

#### Memory Access

**Host API:**
- NCCL internal kernels handle data movement
- User doesn't touch peer memory
- Optimized for standard collectives

**Device API:**
- User kernel directly reads/writes peer memory
- Full control over access patterns
- Enables custom communication patterns

#### Complexity

**Host API:**
- Simple API: `ncclAllReduce`, `ncclBroadcast`, etc.
- NCCL handles all details
- Limited to standard collectives

**Device API:**
- More complex: barriers, memory ordering, pointer calculations
- User manages synchronization
- Unlimited flexibility

---

### 9.3 Performance Trade-offs

**Host API Advantages:**
- Highly optimized kernels for standard collectives
- Better bandwidth utilization for large transfers
- Ring/tree algorithms for scalability
- Reduced user code complexity

**Device API Advantages:**
- Lower latency for small operations (no host round-trip)
- Fusion of compute and communication in single kernel
- Custom collective patterns not in standard NCCL
- Fine-grained control over synchronization

**Latency Comparison (Approximate):**

```
Host API AllReduce (small message):
  1. CPU calls ncclAllReduce       ~1 μs
  2. Launch NCCL kernel            ~5 μs
  3. Kernel execution              ~10 μs
  4. Return to CPU                 ~1 μs
  Total: ~17 μs

Device API Custom Reduce (small message):
  1. Already in kernel             0 μs
  2. LSA barrier                   ~2 μs
  3. Direct peer access            ~5 μs
  4. LSA barrier                   ~2 μs
  Total: ~9 μs (47% reduction!)
```

**Bandwidth Comparison (Large Message):**

```
Host API AllReduce (1GB):
  - Optimized ring algorithm
  - ~100 GB/s effective bandwidth
  - Uses all NVLink bandwidth

Device API Custom Reduce (1GB):
  - Simple all-to-all pattern
  - ~50 GB/s effective bandwidth
  - Less efficient than optimized algorithm
```

**Recommendation:**
- Use **Host API** for standard collectives and large messages
- Use **Device API** for custom patterns, small frequent communication, or compute-communication fusion

---

## Data Structures

### 10.1 Core Device API Structures

#### ncclWindow_vidmem

**File:** [src/include/nccl_device/impl/core__types.h:13-22](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/core__types.h#L13)

```cpp
struct ncclWindow_vidmem {
  void* winHost;              // Pointer back to host ncclDevrWindow
  char* lsaFlatBase;          // Base of rank 0's memory in flat VA
  int lsaRank;                // This rank's LSA rank
  int worldRank;              // This rank's world rank
  uint32_t stride4G;          // Stride between ranks (in 4GB units)
  uint32_t mcOffset4K;        // Multicast offset (in 4KB units)
  uint32_t ginOffset4K;       // GIN offset (in 4KB units)
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONTEXTS];
};
```

**Purpose:** Device-side window metadata for pointer calculations

**Key Calculations:**
- Local pointer: `lsaFlatBase + lsaRank * (stride4G * 4GB) + offset`
- Peer pointer: `lsaFlatBase + peer * (stride4G * 4GB) + offset`
- Multicast pointer: `mcBasePtr + (mcOffset4K * 4KB) + offset`

#### ncclDevrWindow

**File:** [src/include/dev_runtime.h:20-28](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/dev_runtime.h#L20)

```cpp
struct ncclDevrWindow {
  struct ncclDevrMemory* memory;   // Underlying memory allocation
  void* userPtr;                   // User-visible pointer
  size_t size;                     // Window size
  size_t bigOffset;                // Offset in symmetric VA space
  int winFlags;                    // Window flags (SYMMETRIC, etc.)
  void* localRegHandle;            // Local registration handle
  struct ncclWindow_vidmem* vidmem; // Device-side metadata
};
```

**Purpose:** Host-side window tracking

#### ncclDevrMemory

**File:** [src/dev_runtime.cc:17-26](/home/jeromeku/torchcomms/thirdparty/nccl/src/dev_runtime.cc#L17)

```cpp
struct ncclDevrMemory {
  int refCount;                           // Reference count
  struct ncclDevrMemory* next;            // Linked list
  CUmemGenericAllocationHandle memHandle; // CUDA memory handle
  void* primaryAddr;                      // Primary VA for this memory
  size_t size;                            // Memory size
  size_t bigOffset;                       // Offset in symmetric VA space
  void* ginHostWins[NCCL_GIN_MAX_CONTEXTS]; // GIN host handles
  ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONTEXTS]; // GIN device handles
};
```

**Purpose:** Track underlying memory allocations and their symmetric mappings

---

### 10.2 Barrier Structures

#### ncclLsaBarrierHandle

**File:** [src/include/nccl_device/impl/lsa_barrier__types.h:12-15](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/lsa_barrier__types.h#L12)

```cpp
struct ncclLsaBarrierHandle {
  ncclDevResourceHandle_t bufHandle;  // Handle to barrier state buffer
  int nBarriers;                      // Number of barriers
};
```

#### ncclLsaBarrierSession

**File:** [src/include/nccl_device/impl/lsa_barrier__types.h:18-43](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/impl/lsa_barrier__types.h#L18)

```cpp
template<typename Coop>
struct ncclLsaBarrierSession_internal {
  Coop coop;                    // Cooperation scope
  ncclDevComm const& comm;      // Device communicator
  ncclTeam team;                // Team being synchronized
  ncclLsaBarrierHandle handle;  // Barrier handle
  int index;                    // Barrier index
  bool multimem;                // Using multicast?
  ncclMultimemHandle mmHandle;  // Multicast handle
  uint32_t epoch;               // Current epoch

  // Inbox pointer calculations
  uint32_t* mcInbox(bool multimem);
  uint32_t* ucInbox(int owner, int peer);
};
```

---

### 10.3 Cooperation Scope Structures

**File:** [src/include/nccl_device/coop.h](/home/jeromeku/torchcomms/thirdparty/nccl/src/include/nccl_device/coop.h)

```cpp
// Single thread
typedef ncclCoopTile<1> ncclCoopThread;

// Full warp (32 threads)
typedef ncclCoopTile<32> ncclCoopWarp;

// Subset of warp lanes
struct ncclCoopLanes {
  uint32_t lmask;  // Lane mask
  // ...
};

// Multiple consecutive warps
struct ncclCoopWarpSpan {
  uint32_t warp0:8, nWarps:8, id:8;
  // ...
};

// Entire CTA
struct ncclCoopCta {
  int thread_rank() const { return threadIdx.x; }
  int size() const { return blockDim.x; }
  void sync() { __syncthreads(); }
};
```

**All cooperation types provide:**
- `thread_rank()`: Thread's rank within group
- `size()`: Number of threads in group
- `sync()`: Synchronize all threads in group

---

## Sequence Diagrams

### 11.1 Host-Side Setup Flow

```
User Application          ncclCommWindowRegister      symMemoryObtain        symMemoryMapLsaTeam
     |                            |                         |                        |
     |--ncclCommWindowRegister--->|                         |                        |
     |    (send_win)              |                         |                        |
     |                            |--cuMemGetAddressRange-->|                        |
     |                            |<------------------------|                        |
     |                            |                         |                        |
     |                            |--cuMemRetainHandle----->|                        |
     |                            |<------------------------|                        |
     |                            |                         |                        |
     |                            |--symMemoryObtain------->|                        |
     |                            |                         |--Export Handle-------->|
     |                            |                         |                        |
     |                            |                         |   [Bootstrap AllGather]|
     |                            |                         |   (Exchange Handles)   |
     |                            |                         |                        |
     |                            |                         |   [For Each Peer]      |
     |                            |                         |--Import Handle-------->|
     |                            |                         |--cuMemMap------------->|
     |                            |                         |--cuMemSetAccess------->|
     |                            |                         |<-----------------------|
     |                            |                         |                        |
     |                            |                         |   [Bootstrap Barrier]  |
     |                            |<------------------------|                        |
     |                            |                         |                        |
     |                            |--symWindowCreate------->|                        |
     |                            |  (Create device struct) |                        |
     |                            |<------------------------|                        |
     |                            |                         |                        |
     |                            |   [Bootstrap Barrier]   |                        |
     |<---------------------------|                         |                        |
     |                            |                         |                        |
     |--ncclDevCommCreate-------->|                         |                        |
     |                            |--Allocate Barriers----->|                        |
     |                            |--Allocate Resources---->|                        |
     |<---------------------------|                         |                        |
     |                            |                         |                        |
     |--Launch Kernel------------>|                         |                        |
```

---

### 11.2 Device Kernel Execution with Barriers

```
GPU 0 CTA 5              GPU 1 CTA 5              GPU 2 CTA 5              GPU 3 CTA 5
    |                        |                        |                        |
    |--Barrier Session------>|                        |                        |
    |  Constructor           |--Barrier Session------>|                        |
    |  (Load Epoch)          |  Constructor           |--Barrier Session------>|
    |                        |  (Load Epoch)          |  Constructor           |
    |                        |                        |  (Load Epoch)          |
    |                        |                        |                        |
    |--sync(relaxed)-------->|--sync(relaxed)-------->|--sync(relaxed)-------->|
    |  [ENTRY BARRIER]       |  [ENTRY BARRIER]       |  [ENTRY BARRIER]       |
    |                        |                        |                        |
    |  arrive():             |  arrive():             |  arrive():             |
    |    Write→1,2,3         |    Write→0,2,3         |    Write→0,1,3         |
    |    ----------------->  |    <--------------->   |    <--------------->   |
    |    --------------------------->                 |                        |
    |    ------------------------------------------------>                     |
    |                        |                        |                        |
    |  wait():               |  wait():               |  wait():               |
    |    Poll inbox 1,2,3    |    Poll inbox 0,2,3    |    Poll inbox 0,1,3    |
    |    (Spin until all=1)  |    (Spin until all=1)  |    (Spin until all=1)  |
    |                        |                        |                        |
    |  [All Arrived]         |  [All Arrived]         |  [All Arrived]         |
    |                        |                        |                        |
    |--Computation---------->|--Computation---------->|--Computation---------->|
    |  For each offset:      |  For each offset:      |  For each offset:      |
    |    For peer in 0..3:   |    For peer in 0..3:   |    For peer in 0..3:   |
    |      ptr=GetLsaPtr()   |      ptr=GetLsaPtr()   |      ptr=GetLsaPtr()   |
    |      v += load(ptr)    |      v += load(ptr)    |      v += load(ptr)    |
    |    For peer in 0..3:   |    For peer in 0..3:   |    For peer in 0..3:   |
    |      ptr=GetLsaPtr()   |      ptr=GetLsaPtr()   |      ptr=GetLsaPtr()   |
    |      store(ptr, v)     |      store(ptr, v)     |      store(ptr, v)     |
    |      -------------->   |      <------------->   |      <------------->   |
    |      ---------------------->                    |                        |
    |      ---------------------------------------------->                     |
    |                        |                        |                        |
    |--sync(release)-------->|--sync(release)-------->|--sync(release)-------->|
    |  [EXIT BARRIER]        |  [EXIT BARRIER]        |  [EXIT BARRIER]        |
    |  (Fence all stores)    |  (Fence all stores)    |  (Fence all stores)    |
    |                        |                        |                        |
    |  arrive():             |  arrive():             |  arrive():             |
    |    Write→1,2,3         |    Write→0,2,3         |    Write→0,1,3         |
    |    ----------------->  |    <--------------->   |    <--------------->   |
    |                        |                        |                        |
    |  wait():               |  wait():               |  wait():               |
    |    Poll inbox 1,2,3    |    Poll inbox 0,2,3    |    Poll inbox 0,1,3    |
    |    (Spin until all=2)  |    (Spin until all=2)  |    (Spin until all=2)  |
    |                        |                        |                        |
    |  [All Completed]       |  [All Completed]       |  [All Completed]       |
    |                        |                        |                        |
    |--Barrier Destructor--->|--Barrier Destructor--->|--Barrier Destructor--->|
    |  (Save Epoch=2)        |  (Save Epoch=2)        |  (Save Epoch=2)        |
    |                        |                        |                        |
    |--Return--------------->|--Return--------------->|--Return--------------->|
```

---

### 11.3 Memory Access Pattern for AllReduce

```
Example: 4 ranks, element at offset 0

Initial State:
  Rank 0: sendbuff[0] = 0.0
  Rank 1: sendbuff[0] = 1.0
  Rank 2: sendbuff[0] = 2.0
  Rank 3: sendbuff[0] = 3.0

Phase 1: Reduce (each rank reads from all peers)
  Rank 0 Thread:
    v = 0
    v += GetLsaPtr(sendwin, 0, 0)[0]  -> v = 0.0
    v += GetLsaPtr(sendwin, 0, 1)[0]  -> v = 1.0 (RDMA read)
    v += GetLsaPtr(sendwin, 0, 2)[0]  -> v = 3.0 (RDMA read)
    v += GetLsaPtr(sendwin, 0, 3)[0]  -> v = 6.0 (RDMA read)
    Result: v = 6.0

  Rank 1 Thread:
    v = 0
    v += GetLsaPtr(sendwin, 0, 0)[0]  -> v = 0.0 (RDMA read)
    v += GetLsaPtr(sendwin, 0, 1)[0]  -> v = 1.0
    v += GetLsaPtr(sendwin, 0, 2)[0]  -> v = 3.0 (RDMA read)
    v += GetLsaPtr(sendwin, 0, 3)[0]  -> v = 6.0 (RDMA read)
    Result: v = 6.0

  [Same for Ranks 2 and 3]

Phase 2: Broadcast (each rank writes to all peers)
  Rank 0 Thread:
    GetLsaPtr(recvwin, 0, 0)[0] = 6.0  (local write)
    GetLsaPtr(recvwin, 0, 1)[0] = 6.0  (RDMA write)
    GetLsaPtr(recvwin, 0, 2)[0] = 6.0  (RDMA write)
    GetLsaPtr(recvwin, 0, 3)[0] = 6.0  (RDMA write)

  [All ranks write same value to all peers]

Final State (after exit barrier):
  Rank 0: recvbuff[0] = 6.0
  Rank 1: recvbuff[0] = 6.0
  Rank 2: recvbuff[0] = 6.0
  Rank 3: recvbuff[0] = 6.0
```

**Memory Access Pattern Visualization:**

```
           Rank 0         Rank 1         Rank 2         Rank 3
           +-----+        +-----+        +-----+        +-----+
Send       | 0.0 |        | 1.0 |        | 2.0 |        | 3.0 |
           +-----+        +-----+        +-----+        +-----+
              ^              ^              ^              ^
              |              |              |              |
              |  +----------++--------------++-----------+ |
              |  |           |               |           | |
              |  |  +--------+--------------++---------+ | |
              |  |  |        |              |          | | |
              |  |  |  +-----+-------------++---------+| | |
              |  |  |  |     |             |          || | |
           Read All -------------------------------------- Read All
              |  |  |  |     |             |          || | |
              v  v  v  v     v             v          vv v v
           +-----+        +-----+        +-----+        +-----+
Compute    | 6.0 |        | 6.0 |        | 6.0 |        | 6.0 |
           +-----+        +-----+        +-----+        +-----+
              |              |              |              |
              +--+--------+--+--+--------+--+--+--------+--+
                 |        |     |        |     |        |
              +--+        +--+--+        +--+--+        +--+
              |              |              |              |
              v              v              v              v
           +-----+        +-----+        +-----+        +-----+
Recv       | 6.0 |        | 6.0 |        | 6.0 |        | 6.0 |
           +-----+        +-----+        +-----+        +-----+
         Write All                                    Write All
```

---

## Performance Characteristics

### 12.1 Latency Analysis

**Components of Device API Latency:**

1. **Barrier Synchronization:**
   - Entry barrier: ~2 μs
   - Exit barrier: ~2 μs
   - Total: ~4 μs overhead

2. **Peer Memory Access:**
   - Local read: ~100 ns
   - Remote read (NVLink): ~500 ns
   - Remote write (NVLink): ~800 ns

3. **Computation:**
   - Floating-point add: ~1 cycle (~1 ns)
   - Per-element cost: negligible

**Total Latency for Small AllReduce (1KB, 4 ranks):**

```
Entry barrier:              2 μs
Read from 4 peers:          4 * 500 ns = 2 μs
Reduction computation:      < 1 μs
Write to 4 peers:           4 * 800 ns = 3.2 μs
Exit barrier:               2 μs
Total:                      ~10.2 μs
```

**Comparison with Host API (1KB):**

```
Host API:
  ncclAllReduce call:       1 μs
  Kernel launch:            5 μs
  Kernel execution:         10 μs
  Return to host:           1 μs
  Total:                    ~17 μs

Device API Advantage:       ~40% lower latency
```

---

### 12.2 Bandwidth Analysis

**Peak Theoretical Bandwidth:**

- NVLink 4.0: 900 GB/s bidirectional per GPU
- PCIe 5.0: 128 GB/s bidirectional
- For 8 GPUs with NVLink: ~7.2 TB/s aggregate

**Achievable Bandwidth (AllReduce, 1GB, 8 ranks):**

**Host API (Ring Algorithm):**
```
Ring AllReduce:
  Step 1: ReduceScatter (N-1 steps)
  Step 2: AllGather (N-1 steps)

  Each step transfers: 1GB / N = 128 MB
  Total steps: 2(N-1) = 14
  Link bandwidth: 900 GB/s

  Time = (2(N-1) * (1GB/N)) / BW
       = (14 * 128 MB) / 900 GB/s
       = ~2 ms

  Effective BW = 1GB / 2ms = 500 GB/s
```

**Device API (Naive All-to-All):**
```
All-to-All Pattern:
  Each rank reads from N-1 peers: (N-1) * 1GB = 7 GB
  Each rank writes to N-1 peers: (N-1) * 1GB = 7 GB

  Network contention factor: ~2-3x

  Time = (7 GB) / (900 GB/s / 2.5)
       = ~20 ms

  Effective BW = 1GB / 20ms = 50 GB/s
```

**Conclusion:** Host API is 10x better for large messages due to optimized ring algorithm.

---

### 12.3 Scalability Characteristics

**LSA Barrier Scalability:**

| LSA Team Size | Barrier Latency | Notes |
|---------------|-----------------|-------|
| 2 GPUs | ~1.5 μs | Minimal contention |
| 4 GPUs | ~2.0 μs | Moderate contention |
| 8 GPUs | ~3.0 μs | Higher contention |
| 16 GPUs | ~5.0 μs | Significant contention |

**Barrier latency grows with team size due to:**
1. More peers to signal/wait for
2. Atomic contention on inboxes
3. Memory bandwidth limits

**Recommendation:** Keep LSA teams small (≤8 GPUs) for best latency.

---

### 12.4 When to Use Device API

**Use Device API when:**

✅ **Small, frequent communication** (< 1 MB, many operations)
- Device API latency advantage overcomes bandwidth disadvantage
- Example: Gradient averaging in distributed training (per-layer)

✅ **Compute-communication fusion**
- Single kernel does both computation and communication
- Example: Halo exchange in stencil computations

✅ **Custom communication patterns**
- Patterns not available in standard NCCL
- Example: Sparse all-to-all, selective broadcast

✅ **Intra-node only** (LSA team ≤ 8 GPUs)
- Device API optimized for peer-to-peer within node
- Example: Model parallelism within node

**Use Host API when:**

✅ **Large messages** (> 1 MB)
- Optimized ring/tree algorithms provide better bandwidth
- Example: Full model AllReduce

✅ **Standard collectives**
- AllReduce, Broadcast, AllGather with optimized implementations
- Example: Data parallel training

✅ **Multi-node communication**
- Host API leverages GIN (GPU Initiated Network) efficiently
- Example: Distributed training across nodes

✅ **Simplicity preferred**
- Standard API is easier to use correctly
- Example: Most production workloads

---

### 12.5 Fusion Example: Compute + Communication

**Traditional Approach (Host API):**

```cpp
// Kernel 1: Local computation
localComputeKernel<<<...>>>(input, output, size);

// Kernel 2: NCCL AllReduce (launched internally)
ncclAllReduce(output, output, size, ncclFloat, ncclSum, comm, stream);

// Kernel 3: Continuation
continuationKernel<<<...>>>(output, final_output);

// 3 kernel launches, 2 host synchronizations
```

**Fused Approach (Device API):**

```cpp
// Single kernel: Compute + Communicate + Continue
fusedKernel<<<...>>>(input, final_output, sendwin, recvwin, devComm);

// Inside fusedKernel:
__global__ void fusedKernel(...) {
  // Phase 1: Local computation
  float local = compute(input[tid]);

  // Phase 2: AllReduce using Device API
  bar.sync(relaxed);
  float reduced = 0;
  for (int peer = 0; peer < nRanks; peer++) {
    float* ptr = (float*)ncclGetLsaPointer(sendwin, 0, peer);
    reduced += ptr[tid];
  }
  bar.sync(release);

  // Phase 3: Continuation
  final_output[tid] = continue_compute(reduced);
}

// 1 kernel launch, no host synchronization
```

**Performance Comparison:**

```
Traditional:
  Kernel 1 launch:     5 μs
  Kernel 1 execute:    100 μs
  Kernel 2 launch:     5 μs
  Kernel 2 execute:    50 μs
  Kernel 3 launch:     5 μs
  Kernel 3 execute:    100 μs
  Total:               265 μs

Fused:
  Kernel launch:       5 μs
  Phase 1 execute:     100 μs
  Barrier + Reduce:    10 μs
  Phase 3 execute:     100 μs
  Total:               215 μs

Speedup:               ~23% (265/215)
```

**Key Benefits:**
- Eliminates 2 kernel launches (~10 μs)
- Eliminates host-device synchronization overhead
- Better cache locality (data stays warm)
- Reduced scheduling overhead

---

## Summary

NCCL's Device API provides a powerful mechanism for GPU kernels to perform collective communication directly, without CPU intervention. The key components are:

1. **Symmetric Memory Windows**: Enable predictable peer memory access via flat VA space
2. **LSA Barriers**: Cross-GPU synchronization using epoch-based atomic protocols
3. **Device Communicator**: Encapsulates rank info, barriers, and resource buffers
4. **Direct Peer Access**: `ncclGetLsaPointer` calculates addresses in symmetric VA space

**When to Use:**
- Small, frequent communication (< 1 MB)
- Compute-communication fusion
- Custom collective patterns
- Intra-node only (≤ 8 GPUs)

**Trade-offs:**
- Lower latency than Host API for small operations
- Lower bandwidth than Host API for large operations
- More complex programming model
- Full flexibility for custom patterns

The Device API is a specialized tool for performance-critical applications that need fine-grained control over GPU-to-GPU communication. For most applications, the standard Host API remains the recommended choice.

---

**End of Trace**
