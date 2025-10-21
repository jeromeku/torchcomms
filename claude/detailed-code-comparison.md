# Detailed Code Comparison: NCCLX vs Upstream NCCL

**Companion Document to:** ncclx-vs-nccl-analysis.md
**Focus:** Detailed source code differences with full snippets

---

## Table of Contents

1. [Communicator Structure Differences](#communicator-structure-differences)
2. [Task Structure Evolution](#task-structure-evolution)
3. [Symmetric Kernel Implementations](#symmetric-kernel-implementations)
4. [GIN Implementation Details](#gin-implementation-details)
5. [Copy Engine Implementation](#copy-engine-implementation)
6. [Transport Layer Differences](#transport-layer-differences)

---

## Communicator Structure Differences

### Core Communicator Extensions

**File:** `src/include/comm.h`

#### Unified Diff of Key Sections

```diff
--- comms/ncclx/v2_27/src/include/comm.h
+++ thirdparty/nccl/src/include/comm.h
@@ -17,27 +18,9 @@
 #include "graph.h"
 #include "profiler.h"
 #include "allocator.h"
-
-#include <optional>
-
-#include "comms/ctran/CtranComm.h"
-#include "comms/utils/colltrace/CollTraceInterface.h"
-#include "comms/ctran/memory/SlabAllocator.h"
-#include "comms/ctran/memory/memCacheAllocator.h"
-#include "comms/utils/commSpecs.h"
-
-// Forward declarations of ncclx classes
-class ICtran;
-namespace ctran::bootstrap {
-class IBootstrap;
-}
-class CollTrace;
-namespace ncclx {
-class CommStateX;
-}
-namespace ncclx::transport {
-class TransportProxy;
-}
+#include "dev_runtime.h"
+#include "sym_kernels.h"
+#include "ce_coll.h"
```

**Analysis:**

- **NCCLX** includes Meta-specific infrastructure headers:
  - `CtranComm.h`: Communication transformation layer
  - `CollTraceInterface.h`: Tracing/profiling
  - `SlabAllocator.h`, `memCacheAllocator.h`: Memory management
  - Forward declarations for Meta-internal classes

- **Upstream NCCL** includes new feature headers:
  - `dev_runtime.h`: Device runtime API
  - `sym_kernels.h`: Symmetric kernel definitions
  - `ce_coll.h`: Copy Engine collectives

#### Shared Resource Structure Diff

```diff
@@ -157,6 +140,9 @@

   /* proxy related shared res */
   struct ncclProxyState* proxyState;
+
+  // GIN state
+  struct ncclGinState ginState;
 };
```

**New GIN State in Upstream NCCL:**

From [`thirdparty/nccl/src/include/gin/gin_host.h:16-36`](thirdparty/nccl/src/include/gin/gin_host.h#L16-L36):

```cpp
struct ncclGinState {
  ncclGin_t* ncclGin;                           // Plugin-provided GIN interface
  void* ginInstance;                             // Opaque GIN instance handle
  bool connected;                                // GIN connection established
  int ginType;                                   // GIN implementation type identifier
  int ginCommCount;                              // Active GIN communicators
  void* ginComms[NCCL_GIN_MAX_CONTEXTS];        // GIN communicator handles per context
  void* ginCtx[NCCL_GIN_MAX_CONTEXTS];          // GIN context handles
  ncclNetDeviceHandle_t* ginDevHandles[NCCL_GIN_MAX_CONTEXTS];  // Device handles

  int needsProxyProgress;                        // Whether proxy must call progress
  int ginProgress;                               // GIN progress enabled flag

  // Background progress thread
  pthread_t thread;                              // Progress thread handle
  pthread_mutex_t threadLock;                    // Thread synchronization mutex
  pthread_cond_t threadCond;                     // Thread condition variable
  ncclResult_t asyncResult;                      // Async operation result

  // Signal/counter memory management
  int signalSpaceSize;                           // Total signal space allocated
  int counterSpaceSize;                          // Total counter space allocated
  ncclSpace signalSpace;                         // Signal memory region descriptor
  ncclSpace counterSpace;                        // Counter memory region descriptor
};
```

**GIN API Functions:**

From [`thirdparty/nccl/src/include/gin/gin_host.h:41-52`](thirdparty/nccl/src/include/gin/gin_host.h#L41-L52):

```cpp
// Initialize GIN once per communicator
// Establishes connection, allocates resources
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm);

// Cleanup all GIN resources
ncclResult_t ncclGinFinalize(struct ncclComm* comm);

// Progress GIN operations (non-blocking poll)
// Must be called regularly for async progress
ncclResult_t ncclGinProgress(struct ncclGinState* ginState);

// Register memory region with GIN
// Creates both host and device windows for GPU-initiated access
ncclResult_t ncclGinRegister(struct ncclComm* comm,
                             void* address, size_t size,
                             void* ginHostWins[NCCL_GIN_MAX_CONTEXTS],
                             ncclGinWindow_t ginDevWins[NCCL_GIN_MAX_CONTEXTS]);

// Deregister previously registered memory
ncclResult_t ncclGinDeregister(struct ncclComm* comm,
                               void* ginHostWins[NCCL_GIN_MAX_CONTEXTS]);

// Allocate signal and counter resources for synchronization
ncclResult_t ncclGinAllocSignalsCounters(struct ncclComm* comm,
                                         int nSignals, uint32_t* outSignal0,
                                         int nCounters, uint32_t* outCounter0);

// Free signal and counter resources
ncclResult_t ncclGinFreeSignalsCounters(struct ncclComm* comm,
                                        uint32_t signal0, int nSignals,
                                        uint32_t counter0, int nCounters);

// Query for errors in GIN operations
ncclResult_t ncclGinQueryLastError(struct ncclGinState* ginState, bool* hasError);
```

---

## Task Structure Evolution

### Collective Task Extensions

**File:** `src/include/comm.h`

```diff
@@ -218,12 +204,14 @@
   int32_t nMaxChannels:8;
   int32_t nWarps:8;
   int32_t algorithm:8, protocol:8;
-  uint32_t isCollnet:1, isNvls:1;
-  uint32_t devFuncId:30;
+  uint32_t isCollnet:1, isNvls:1, isSymLast:1;
+  uint32_t devFuncId:29;
   int regBufType;
   // number of elements in planner->ipcMemQueue associated with this collective
   int nCleanupQueueElts;

+  struct ncclDevrWindow* sendWin;
+  struct ncclDevrWindow* recvWin;
   void* sendMhandle;
   void* recvMhandle;
   void** sendNetHandles;
@@ -237,12 +225,16 @@

   // Profiler plugin
   int eActivationMask;
+  void* groupApiEventHandle;
+  void* collApiEventHandle;
   void* eventHandle;
   uint8_t nChannels;
 };
```

**Analysis:**

1. **Window Pointers Added (Upstream NCCL):**
   ```cpp
   struct ncclDevrWindow* sendWin;  // Send buffer window registration
   struct ncclDevrWindow* recvWin;  // Receive buffer window registration
   ```
   These enable GIN and CE to access registered memory with device-side metadata.

2. **Symmetric Kernel Tracking:**
   ```cpp
   uint32_t isSymLast:1;  // Indicates this is the last task in a symmetric batch
   ```

3. **Enhanced Profiler Events:**
   ```cpp
   void* groupApiEventHandle;   // Group-level event for ncclGroupStart/End
   void* collApiEventHandle;    // Collective API call event
   ```

### P2P Task Extensions

```diff
@@ -250,6 +243,8 @@
 struct ncclTaskP2p {
   struct ncclTaskP2p* next;
   ncclFunc_t func;
+  ncclFunc_t collAPI;
   void* buff;
   size_t count;
   ncclDataType_t datatype;
@@ -257,6 +252,8 @@

   // Profiler plugin
   int eActivationMask;
+  void* groupApiEventHandle;
+  void* p2pApiEventHandle;
   void* eventHandle;
   uint8_t nChannels;
 };
```

**New Fields:**

```cpp
ncclFunc_t collAPI;             // Tracks if P2P is part of collective pattern
void* groupApiEventHandle;      // Group-level profiler event
void* p2pApiEventHandle;        // P2P-specific profiler event
```

### Task Group Extensions

```diff
@@ -266,12 +260,14 @@
   bool persistent; // aka captured in a graph
   bool isHostCbEnq;
   bool isSymColl;
+  bool isCeColl;
   enum ncclDevWorkStorageType workStorageType;
```

**New Flag:**

```cpp
bool isCeColl;  // Task uses Copy Engine collectives
```

---

## Symmetric Kernel Implementations

### NCCLX Symmetric Kernels

**File:** [`comms/ncclx/v2_27/src/include/symmetric.h`](comms/ncclx/v2_27/src/include/symmetric.h)

**Device Base Structure:**

From [`comms/ncclx/v2_27/src/include/symmetric.h:20-31`](comms/ncclx/v2_27/src/include/symmetric.h#L20-L31):

```cpp
struct alignas(16) ncclSymDevBase {
  uint32_t llEpoch[ncclSymMaxBlocks];             // LL protocol epoch per block
  uint32_t barEpochMc[ncclSymMaxBlocks];          // Barrier epoch (multicast coherent)
  uint32_t barEpochUc[ncclSymMaxBlocks];          // Barrier epoch (uncached)
  uint32_t barInboxMc[ncclSymMaxBlocks];          // Barrier inbox (multicast)
  uint32_t barInboxPerPeer[];                     // Variable-length per-peer inboxes

  // Calculate total size including variable-length arrays
  static constexpr size_t size(int nRanks) {
    return sizeof(ncclSymDevBase) +
           alignUp(ncclSymMaxBlocks*nRanks*sizeof(uint32_t), 16) +
           ncclSymMaxBlocks * /*epochs=*/2 * ncclSymLLEpochSize(nRanks);
  }
};
```

**Device Communicator:**

From [`comms/ncclx/v2_27/src/include/symmetric.h:45-51`](comms/ncclx/v2_27/src/include/symmetric.h#L45-L51):

```cpp
struct ncclSymDevComm {
  ncclSymDevBase* base;        // Base pointer (UC memory)
  ncclSymDevBase* baseMc;      // Base pointer (MC memory, cache-coherent)
  uint32_t stride4G;           // 4GB stride for address calculation
  int nRanks, rank;            // Communicator size and this rank
  uint32_t nRanks_rcp32;       // Reciprocal for fast division: idivRcp32(nRanks)
};
```

**Kernel Arguments:**

From [`comms/ncclx/v2_27/src/include/symmetric.h:53-60`](comms/ncclx/v2_27/src/include/symmetric.h#L53-L60):

```cpp
struct alignas(16) ncclSymDevArgs {
  struct ncclSymDevComm comm;  // Device communicator
  int rootRank;                // Root rank for rooted operations
  uint64_t redOpArg;           // Reduction operation argument (collectively uniform)
  size_t nElts;                // Number of elements
  char* input;                 // Input buffer pointer
  char* output;                // Output buffer pointer
};
```

**Kernel Enumeration:**

From [`comms/ncclx/v2_27/src/include/symmetric.h:62-78`](comms/ncclx/v2_27/src/include/symmetric.h#L62-L78):

```cpp
enum ncclSymKernelId {
  ncclSymKernelId_AllReduce_AGxLL_R,        // AllReduce: AllGather + LL + Reduce
  ncclSymKernelId_AllReduce_AGxLLMC_R,      // AllReduce: AG + LLMC + Reduce
  ncclSymKernelId_AllReduce_RSxLD_AGxST,    // AllReduce: RS + Load/Direct + AG + Store
  ncclSymKernelId_AllReduce_RSxLDMC_AGxSTMC,// AllReduce: RS + LDMC + AG + STMC

  ncclSymKernelId_AllGather_LL,             // AllGather: Low-Latency
  ncclSymKernelId_AllGather_LLMC,           // AllGather: LL + Multicast
  ncclSymKernelId_AllGather_ST,             // AllGather: Store
  ncclSymKernelId_AllGather_STMC,           // AllGather: Store + Multicast

  ncclSymKernelId_ReduceScatter_LL,         // ReduceScatter: Low-Latency
  ncclSymKernelId_ReduceScatter_LD,         // ReduceScatter: Load/Direct
  ncclSymKernelId_ReduceScatter_LDMC,       // ReduceScatter: LD + Multicast

  ncclSymKernelId_Count
};
```

**API Functions:**

From [`comms/ncclx/v2_27/src/include/symmetric.h:80-89`](comms/ncclx/v2_27/src/include/symmetric.h#L80-L89):

```cpp
// Check if symmetric implementation exists for this operation
bool ncclSymImplemented(ncclFunc_t fn, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty);

// Select best kernel and configuration
ncclResult_t ncclSymPickKernel(struct ncclComm* comm,
                               ncclFunc_t fn,
                               int/*ncclDevRedOp_t*/ red,
                               ncclDataType_t ty,
                               size_t nElts,
                               float* estTimeUs,      // Output: estimated time
                               ncclSymKernelId* kernelId,  // Output: selected kernel
                               int* nBlocks,          // Output: block count
                               int* nWarps);          // Output: warp count per block

// Generated kernel table
extern int const ncclSymKernelCount;
extern void* const ncclSymKernelList[];
void* ncclSymGetKernelPtr(ncclSymKernelId kernelId, int red, ncclDataType_t ty);
const char* ncclSymKernelIdToString(int kernelId);
```

### Upstream NCCL Symmetric Kernels (symk)

**File:** [`thirdparty/nccl/src/include/sym_kernels.h`](thirdparty/nccl/src/include/sym_kernels.h)

**Key Constants:**

From [`thirdparty/nccl/src/include/sym_kernels.h:17-25`](thirdparty/nccl/src/include/sym_kernels.h#L17-L25):

```cpp
#define NCCL_SYM_KERNEL_CELL_SIZE 1024  // Minimum work unit (≥16 bytes)

constexpr int ncclSymkMaxBlocks = 64;
constexpr int ncclSymkMaxThreads = 512;
constexpr int ncclSymkLLMaxEltSize = 8;

constexpr __host__ __device__ int ncclSymkLLMaxSlots(int eltSize = ncclSymkLLMaxEltSize) {
  return ncclSymkMaxThreads*ncclSymkLLMaxEltSize/eltSize;
}
```

**Device Communicator (with Device API):**

From [`thirdparty/nccl/src/include/sym_kernels.h:46-49`](thirdparty/nccl/src/include/sym_kernels.h#L46-L49):

```cpp
struct ncclSymkDevComm {
  struct ncclDevComm devComm;           // Unified device API communicator
  struct ncclLLA2AHandle lsaLLA2A;      // Low-latency all-to-all handle
};
```

**Communicator State:**

From [`thirdparty/nccl/src/include/sym_kernels.h:51-54`](thirdparty/nccl/src/include/sym_kernels.h#L51-L54):

```cpp
struct ncclSymkState {
  bool initialized;                     // Lazy initialization flag
  struct ncclSymkDevComm kcomm;         // Device communicator
};
```

**Work Descriptor:**

From [`thirdparty/nccl/src/include/sym_kernels.h:62-69`](thirdparty/nccl/src/include/sym_kernels.h#L62-L69):

```cpp
struct alignas(16) ncclSymkDevWork {
  uint64_t redOpArg;                           // Reduction operation argument
  size_t nElts;                                // Elements in this work unit
  struct ncclWindow_vidmem* inputWin;          // Input window (registered memory)
  struct ncclWindow_vidmem* outputWin;         // Output window (registered memory)
  size_t inputOff;                             // Offset into input window
  size_t outputOff;                            // Offset into output window
  uint64_t rootRank;                           // Root rank (for rooted ops)
  uint64_t sChannelId:16;                      // Starting channel ID
  uint64_t nChannels:16;                       // Number of channels for this work
  uint64_t padding:32;                         // Alignment padding
};
```

**Work Arguments Container:**

From [`thirdparty/nccl/src/include/sym_kernels.h:71-87`](thirdparty/nccl/src/include/sym_kernels.h#L71-L87):

```cpp
struct alignas(16) ncclSymkDevWorkArgs {
  struct ncclSymkDevComm kcomm;         // Device communicator
  int nMaxChannels;                     // Maximum channels available

  // Variable-length arrays follow this structure:
  // channelWorkRange[nChannels];  // Per-channel work range descriptors
  // ncclSymkDevWork[nWorks];      // Array of work descriptors

  // Helper functions for accessing variable-length data
  __host__ static constexpr size_t calcArgsSize(int nChannels, int nWorks) {
    return alignUp(sizeof(struct ncclSymkDevWorkArgs), 16) +
           alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16) +
           nWorks * sizeof(struct ncclSymkDevWork);
  }

  __host__ __device__ struct ncclSymkChannelWorkRange* getWorkRange() const {
    return (struct ncclSymkChannelWorkRange*)((uint8_t*)this +
           alignUp(sizeof(struct ncclSymkDevWorkArgs), 16));
  }

  __host__ __device__ struct ncclSymkDevWork* getWorks(int nChannels) const {
    return (struct ncclSymkDevWork*)((uint8_t*)this->getWorkRange() +
           alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16));
  }
};
```

**Kernel Enumeration (Extended):**

From [`thirdparty/nccl/src/include/sym_kernels.h:27-44`](thirdparty/nccl/src/include/sym_kernels.h#L27-L44):

```cpp
enum ncclSymkKernelId {
  ncclSymkKernelId_AllReduce_AGxLL_R,
  ncclSymkKernelId_AllReduce_AGxLLMC_R,
  ncclSymkKernelId_AllReduce_RSxLD_AGxST,
  ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC,
  ncclSymkKernelId_AllReduce_RSxNet_ARxMC_AGxNet,  // NEW: Network-aware variant

  ncclSymkKernelId_AllGather_LL,
  ncclSymkKernelId_AllGather_LLMC,
  ncclSymkKernelId_AllGather_ST,
  ncclSymkKernelId_AllGather_STMC,

  ncclSymkKernelId_ReduceScatter_LL,
  ncclSymkKernelId_ReduceScatter_LD,
  ncclSymkKernelId_ReduceScatter_LDMC,

  ncclSymkKernelId_Count
};
```

**API Functions:**

From [`thirdparty/nccl/src/include/sym_kernels.h:95-111`](thirdparty/nccl/src/include/sym_kernels.h#L95-L111):

```cpp
// Initialize symmetric kernel state (lazy, once per comm)
ncclResult_t ncclSymkInitOnce(struct ncclComm* comm);

// Cleanup symmetric kernel resources
ncclResult_t ncclSymkFinalize(struct ncclComm* comm);

// Check if symmetric kernel is available for this operation
bool ncclSymkAvailable(struct ncclComm* comm,
                       ncclFunc_t coll,
                       int/*ncclDevRedOp_t*/ red,
                       ncclDataType_t ty,
                       size_t nElts);

// Pick optimal kernel for potentially batched operations
ncclResult_t ncclSymkPickKernel(struct ncclComm* comm,
                                ncclFunc_t coll,
                                int/*ncclDevRedOp_t*/ red,
                                ncclDataType_t ty,
                                size_t nEltsTotal,     // Total elements across all works
                                size_t nEltsMax,       // Max elements in any single work
                                int nWorks,            // Number of work units
                                float* estTimeUs,      // Output: estimated time
                                ncclSymkKernelId* kernelId,  // Output: kernel ID
                                int* nBlocks,          // Output: block count
                                int* nWarps);          // Output: warp count

// Build device work descriptor from task
ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm,
                                 struct ncclTaskColl* task,
                                 struct ncclSymkDevWork* outDevWork);

// Generated kernel table
extern int const ncclSymkKernelCount;
extern void* ncclSymkKernelList[];
extern int ncclSymkKernelRequirements[/*ncclSymkKernelCount*/];
void* ncclSymkGetKernelPtr(ncclSymkKernelId kernelId, int red, ncclDataType_t ty);
const char* ncclSymkKernelIdToString(int kernelId);
```

---

## GIN Implementation Details

### GIN Proxy Interface

**File:** [`thirdparty/nccl/src/include/gin/gin_host_proxy.h`](thirdparty/nccl/src/include/gin/gin_host_proxy.h#L1-L29)

```cpp
/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#ifndef GIN_HOST_PROXY_H_
#define GIN_HOST_PROXY_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>
#include "nccl.h"
#include "gin/gin_host.h"
#include "plugin/nccl_net.h"

// Create a GIN context via proxy thread
// Used when GIN operations need to be mediated by a separate thread
ncclResult_t ncclGinProxyCreateContext(struct ncclComm *comm,
                                       void *collComm,    // Collective communicator
                                       int devId,         // Device ID
                                       int nSignals,      // Number of signals to allocate
                                       int nCounters,     // Number of counters to allocate
                                       void **outGinCtx,  // Output: GIN context handle
                                       ncclNetDeviceHandle_v11_t **outDevHandle); // Output: device handle

// Register memory via proxy
ncclResult_t ncclGinProxyRegister(ncclGin_t *ginComm,
                                  void *ginCtx,   // GIN context
                                  void *addr,     // Memory address
                                  size_t size,    // Memory size
                                  int type,       // Memory type
                                  int mr_flags,   // Memory region flags
                                  void **mhandle, // Output: memory handle
                                  void **ginHandle); // Output: GIN-specific handle

// Deregister memory via proxy
ncclResult_t ncclGinProxyDeregister(ncclGin_t *ginComm,
                                    void *ginCtx,
                                    void *mhandle);

// Destroy GIN context via proxy
ncclResult_t ncclGinProxyDestroyContext(ncclGin_t *ginComm, void *ginCtx);

// Progress GIN operations via proxy
// Must be called regularly to advance async operations
ncclResult_t ncclGinProxyProgress(ncclGin_t *ginComm, void *ginCtx);

// Query for errors in GIN operations
ncclResult_t ncclGinProxyQueryLastError(ncclGin_t *ginComm,
                                        void *ginCtx,
                                        bool *hasError);

#endif
```

### GDAKI Transport Overview

**Directory:** [`thirdparty/nccl/src/transport/gdaki/`](thirdparty/nccl/src/transport/gdaki/)

**Key Components:**

1. **DOCA GPUNetIO Integration** (`doca-gpunetio/`)
   - Device-side Verbs operations
   - GPU-accessible network primitives
   - RDMA initiation from GPU kernels

2. **GIN Host GDAKI** (`gin_host_gdaki.cc`)
   - Host-side setup and management
   - Context creation and destruction
   - Memory registration with GPUDirect

**File Structure:**

```
thirdparty/nccl/src/transport/gdaki/
├── gin_host_gdaki.cc                    # Host-side GDAKI implementation
├── gin_host_gdaki.h                     # GDAKI host interface
└── doca-gpunetio/
    ├── include/
    │   ├── doca_gpunetio_device.h       # Device-side API
    │   ├── doca_gpunetio_host.h         # Host-side API
    │   ├── doca_gpunetio_config.h       # Configuration
    │   ├── common/
    │   │   ├── doca_gpunetio_verbs_def.h   # Verbs definitions
    │   │   └── doca_gpunetio_verbs_dev.h   # Device Verbs
    │   ├── device/
    │   │   └── [device headers]
    │   └── host/
    │       ├── doca_error.h             # Error handling
    │       ├── doca_gpunetio.h          # Main host API
    │       ├── doca_verbs.h             # Verbs abstraction
    │       ├── doca_gpunetio_high_level.h  # High-level API
    │       ├── mlx5_prm.h               # Mellanox PRM definitions
    │       └── mlx5_ifc.h               # Mellanox interface
    └── src/
        ├── doca_verbs_cuda_wrapper.h    # CUDA integration
        ├── doca_gpunetio_gdrcopy.h      # GDRCopy support
        ├── doca_verbs_mlx5dv_wrapper.h  # MLX5 DV wrapper
        ├── doca_verbs_net_wrapper.h     # Network wrapper
        └── doca_verbs_ibv_wrapper.h     # IB Verbs wrapper
```

---

## Copy Engine Implementation

### CE Collective Header

**File:** [`thirdparty/nccl/src/include/ce_coll.h`](thirdparty/nccl/src/include/ce_coll.h#L1-L77)

**Full Implementation:**

```cpp
/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#ifndef NCCL_CE_COLL_H_
#define NCCL_CE_COLL_H_

#include "nccl.h"
#include "nccl_common.h"
#include "bitops.h"

// Synchronization protocols for Copy Engine operations
// MC (Multicast/Cache-coherent): Fewer operations, requires cache-coherent memory
#define NCCL_CE_SYNC_OPS_PER_RANK_MC 2

// UC (Uncached): More operations, works with any memory type
#define NCCL_CE_SYNC_OPS_PER_RANK_UC 3

// Copy Engine collective state
struct ncclCeColl {
  // Uncached symmetric memory for synchronization
  uint8_t* baseUCSymReadyPtr;                  // Ready signal pointer (UC memory)
  uint8_t* baseUCSymComplPtr;                  // Completion signal pointer (UC memory)
  size_t baseUCSymReadyOffset;                 // Offset to ready signals
  size_t baseUCSymComplOffset;                 // Offset to completion signals

  uint32_t ceSeqNum;                           // Sequence number for ordering
  bool useCompletePtr;                         // Whether to use completion pointer

  // Batching configuration
  uint32_t intraBatchSyncFreq;                 // Sync frequency within batch
  uint64_t intraBatchSyncMsgThreshold;         // Message size threshold for sync

  struct ncclDevrWindow* ceSyncWin;            // Window for CE synchronization
};

// Initialization task for CE setup
struct ncclCeInitTask {
  struct ncclCeInitTask *next;                 // Linked list of init tasks
  struct ncclComm* comm;                       // Associated communicator
};

// Arguments for Copy Engine collective
struct alignas(16) ncclCeCollArgs {
  ncclFunc_t func;                             // Collective function type
  int rootRank;                                // Root rank (for rooted collectives)
  size_t nElts;                                // Number of elements
  size_t eltSize;                              // Element size in bytes
  uint8_t* sendBuff;                           // Send buffer
  uint8_t* recvBuff;                           // Receive buffer
  struct ncclDevrWindow* sendWin;              // Send window (registered memory)
  struct ncclDevrWindow* recvWin;              // Receive window (registered memory)
};

// Batch operation parameters
struct ncclCeBatchOpsParams {
  void** dsts;                                 // Array of destination pointers
  void** srcs;                                 // Array of source pointers
  size_t* sizes;                               // Array of transfer sizes
  size_t numOps;                               // Number of operations in batch
  bool intraBatchSync;                         // Whether to sync within batch
#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes* attrs;                 // Memcpy attributes (for CUDA 12.8+)
  size_t* attrIdxs;                            // Attribute indices
  size_t numAttrs;                             // Number of attributes
#endif
};

// Check if CE implementation exists for this operation
bool ncclCeImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty);

// Initialize CE for a communicator
ncclResult_t ncclCeInit(struct ncclComm* comm);

// Cleanup CE resources
ncclResult_t ncclCeFinalize(struct ncclComm* comm);

// Synchronize memory operations
ncclResult_t ncclMemOpSync(struct ncclComm* comm, cudaStream_t stream);

// Launch CE collective kernel
ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan);

// CE collective implementations
ncclResult_t ncclCeAllGather(struct ncclComm* comm,
                             struct ncclCeCollArgs* args,
                             cudaStream_t stream);

ncclResult_t ncclCeScatter(struct ncclComm* comm,
                           struct ncclCeCollArgs* args,
                           cudaStream_t stream);

ncclResult_t ncclCeGather(struct ncclComm* comm,
                          struct ncclCeCollArgs* args,
                          cudaStream_t stream);

ncclResult_t ncclCeAlltoAll(struct ncclComm* comm,
                            struct ncclCeCollArgs* args,
                            cudaStream_t stream);

#endif /* NCCL_CE_COLL_H_ */
```

### CE AllGather Conceptual Flow

```cpp
// Conceptual implementation of ncclCeAllGather
// (Simplified for illustration)

ncclResult_t ncclCeAllGather(struct ncclComm* comm,
                             struct ncclCeCollArgs* args,
                             cudaStream_t stream) {
  int nRanks = comm->nRanks;
  int myRank = comm->rank;
  size_t chunkSize = args->nElts * args->eltSize;

  // Build batch operation list
  std::vector<void*> dsts(nRanks);
  std::vector<void*> srcs(nRanks);
  std::vector<size_t> sizes(nRanks);

  for (int peer = 0; peer < nRanks; peer++) {
    // Each rank copies its sendbuff to all peers' recvbuff[myRank]
    srcs[peer] = args->sendBuff;
    dsts[peer] = args->recvBuff + peer * chunkSize;  // Peer's receive slot
    sizes[peer] = chunkSize;
  }

  // Execute batch copy using Copy Engine
  ncclCeBatchOpsParams batchParams = {
    .dsts = dsts.data(),
    .srcs = srcs.data(),
    .sizes = sizes.data(),
    .numOps = nRanks,
    .intraBatchSync = (chunkSize > comm->ceColl.intraBatchSyncMsgThreshold)
  };

  // Issue batched memcpy (hardware copy engine handles parallelism)
#if CUDART_VERSION >= 12080
  cudaMemcpyAsync_batched(batchParams.dsts, batchParams.srcs,
                          batchParams.sizes, batchParams.numOps,
                          cudaMemcpyDeviceToDevice, stream);
#else
  // Fallback: issue individual memcpy operations
  for (size_t i = 0; i < batchParams.numOps; i++) {
    cudaMemcpyAsync(batchParams.dsts[i], batchParams.srcs[i],
                    batchParams.sizes[i], cudaMemcpyDeviceToDevice, stream);
  }
#endif

  // Synchronization protocol
  // 1. Write ready signal to UC memory
  uint8_t readySignal = ++comm->ceColl.ceSeqNum;
  uint8_t* myReadyPtr = comm->ceColl.baseUCSymReadyPtr + myRank;
  cudaMemsetAsync(myReadyPtr, readySignal, 1, stream);

  // 2. Wait for all peers' ready signals
  for (int peer = 0; peer < nRanks; peer++) {
    if (peer == myRank) continue;
    uint8_t* peerReadyPtr = comm->ceColl.baseUCSymReadyPtr + peer;

    // Busy-wait polling (on GPU)
    while (*peerReadyPtr != readySignal) {
      __threadfence_system();
    }
  }

  // 3. Optional: completion signal
  if (comm->ceColl.useCompletePtr) {
    uint8_t* myComplPtr = comm->ceColl.baseUCSymComplPtr + myRank;
    cudaMemsetAsync(myComplPtr, readySignal, 1, stream);
  }

  return ncclSuccess;
}
```

---

## Transport Layer Differences

### Standard Transport Files (Both)

Both NCCLX and upstream NCCL share these transport files:

- `net.cc` - Network transport abstraction
- `net_ib.cc` - InfiniBand transport
- `net_socket.cc` - Socket transport
- `p2p.cc` - Peer-to-peer (NVLink/PCIe)
- `shm.cc` - Shared memory transport
- `nvls.cc` - NVSwitch/NVLS transport
- `coll_net.cc` - Collective network transport

### GDAKI Transport (Upstream NCCL Only)

**Unique Directory:** [`thirdparty/nccl/src/transport/gdaki/`](thirdparty/nccl/src/transport/gdaki/)

**Key Innovation:** Direct GPU-to-network I/O using NVIDIA DOCA GPUNetIO

**Conceptual Data Flow:**

```
GPU Kernel Thread
      │
      │ Initiate RDMA Write
      ↓
┌───────────────────────────────────────────────────────┐
│  DOCA GPUNetIO Device API                             │
│  (thirdparty/nccl/src/transport/gdaki/                │
│   doca-gpunetio/include/doca_gpunetio_device.h)       │
├───────────────────────────────────────────────────────┤
│  __device__ void doca_gpu_dev_rdma_write(             │
│    struct doca_gpu_dev_rdma_qp* qp,                   │
│    void* local_addr, size_t length,                   │
│    uint64_t remote_addr, uint32_t rkey) {             │
│                                                        │
│    // Build RDMA WQE (Work Queue Element)             │
│    struct mlx5_wqe_rdma_seg* wqe = get_next_wqe(qp); │
│    wqe->raddr = remote_addr;                          │
│    wqe->rkey = rkey;                                  │
│    wqe->length = length;                              │
│                                                        │
│    // Post directly to NIC's queue                    │
│    __threadfence_system();                            │
│    qp->db_reg = wqe_index;  // Ring doorbell          │
│  }                                                     │
└───────────────────────────────────────────────────────┘
      │
      │ PCIe/NVLink Direct Path
      ↓
┌───────────────────────────────────────────────────────┐
│  ConnectX NIC (GPU-attached)                          │
│  • Directly receives WQE from GPU                     │
│  • No CPU involvement                                 │
│  • Reads source data via GPUDirect                    │
└───────────────────────────────────────────────────────┘
      │
      │ InfiniBand/RoCE Network
      ↓
  Remote NIC → Remote GPU Memory
```

---

## Scheduler Integration (Upstream NCCL Only)

### Scheduler Header

**File:** [`thirdparty/nccl/src/include/scheduler.h`](thirdparty/nccl/src/include/scheduler.h#L1-L18)

```cpp
/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 ************************************************************************/

#ifndef NCCL_SCHEDULER_H_
#define NCCL_SCHEDULER_H_

#include "nccl.h"
#include "comm.h"
#include "sym_kernels.h"

// Analyze task list and separate symmetric-eligible tasks
ncclResult_t ncclMakeSymmetricTaskList(
  struct ncclComm* comm,
  struct ncclTaskColl* task,
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
  struct ncclTaskColl** remainTasksHead);

// Schedule symmetric tasks into optimized kernel plan
ncclResult_t ncclSymmetricTaskScheduler(
  struct ncclComm* comm,
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
  struct ncclKernelPlan* plan);

#endif // NCCL_SCHEDULER_H_
```

### Scheduling Flow Diagram

```
ncclGroupEnd()
      │
      │ Task list built
      ↓
┌───────────────────────────────────────────────────────────┐
│  ncclMakeSymmetricTaskList()                              │
│  (thirdparty/nccl/src/scheduler/symmetric_sched.cc)       │
├───────────────────────────────────────────────────────────┤
│  Input: Task list (ncclTaskColl*)                         │
│                                                            │
│  For each task:                                           │
│    if (ncclSymkAvailable(comm, task->func,                │
│                          task->redOp,                     │
│                          task->dataType,                  │
│                          task->count)) {                  │
│      // Move to symmetric queue                           │
│      symTaskQueue.push(task);                             │
│    } else {                                               │
│      // Keep in standard path                             │
│      *remainTasksHead = append(task);                     │
│    }                                                       │
│                                                            │
│  Output:                                                   │
│    - symTaskQueue: Tasks eligible for symk kernels        │
│    - remainTasksHead: Tasks for standard kernels          │
└───────────────────────────────────────────────────────────┘
      │
      ↓ symTaskQueue not empty
┌───────────────────────────────────────────────────────────┐
│  ncclSymmetricTaskScheduler()                             │
│  (thirdparty/nccl/src/scheduler/symmetric_sched.cc)       │
├───────────────────────────────────────────────────────────┤
│  Input: symTaskQueue                                       │
│                                                            │
│  1. Analyze task batch:                                   │
│     - Compute total data volume                           │
│     - Find max element count                              │
│     - Count number of tasks                               │
│                                                            │
│  2. Select kernel:                                        │
│     ncclSymkPickKernel(comm, func, red, datatype,         │
│                        nEltsTotal, nEltsMax, nTasks,      │
│                        &estTimeUs, &kernelId,             │
│                        &nBlocks, &nWarps);                │
│                                                            │
│  3. Build kernel plan:                                    │
│     - Allocate ncclSymkDevWorkArgs                        │
│     - Fill device communicator                            │
│     - Configure channel work ranges                       │
│     - Populate work descriptors for each task             │
│                                                            │
│  4. Store plan:                                           │
│     plan->symkPlan = {                                    │
│       .kernelPtr = ncclSymkGetKernelPtr(kernelId, ...),   │
│       .args = devWorkArgs,                                │
│       .nBlocks = nBlocks,                                 │
│       .nThreads = nWarps * WARP_SIZE                      │
│     };                                                     │
│                                                            │
│  Output: Populated kernel plan                            │
└───────────────────────────────────────────────────────────┘
      │
      ↓
  Launch Kernel (single launch for entire batch)
```

---

## Summary

This detailed comparison highlights the architectural evolution of upstream NCCL with:

1. **GIN Infrastructure** - Complete GPU-initiated networking stack
2. **GDAKI Transport** - Direct GPU-to-NIC communication
3. **Copy Engine Collectives** - Hardware-accelerated data movement
4. **Advanced Scheduler** - Batch optimization for symmetric operations
5. **Modular Device API** - Extensible device-side interface

NCCLX maintains compatibility while integrating Meta-specific infrastructure (CtranComm, CollTrace, custom allocators).

---

**End of Detailed Code Comparison**

Generated by Claude Code Analysis - 2025-10-21
