# NCCL Architecture

This document describes the architecture of NCCL (NVIDIA Collective Communications Library), following the style of [ARCHITECTURE.md](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html).

## Bird's Eye View

NCCL is a multi-GPU communication library that provides optimized collective operations (AllReduce, Broadcast, AllGather, etc.) and point-to-point primitives. The library orchestrates communication across GPUs using various transport mechanisms (NVLink, PCIe, InfiniBand, Ethernet) with the goal of maximizing bandwidth and minimizing latency.

**The fundamental flow is**: User calls a collective API → NCCL enqueues work into channels → CUDA kernel executes the operation using device primitives → Data moves through optimal transport paths → Proxy threads handle network operations asynchronously.

NCCL operates in layers:
1. **Public API Layer**: User-facing C API ([nccl.h.in](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/nccl.h.in))
2. **Operation Enqueueing**: Planning and scheduling work ([enqueue.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/enqueue.cc))
3. **Transport Layer**: Abstract transport selection and setup ([transport.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/transport.h), [transport/*.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/transport/))
4. **Device Layer**: CUDA kernels and device-side primitives ([device/*.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/))
5. **Proxy Layer**: Asynchronous network progress threads ([proxy.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/proxy.cc))

## Code Layout

```
thirdparty/nccl/
├── src/
│   ├── nccl.h.in                    # Main public API header
│   ├── init.cc                      # Communicator initialization
│   ├── enqueue.cc                   # Operation enqueueing and kernel launch
│   ├── collectives.cc               # Collective API implementations
│   ├── bootstrap.cc                 # Out-of-band initialization protocol
│   ├── proxy.cc                     # Async network proxy threads
│   ├── group.cc                     # Group semantics (ncclGroupStart/End)
│   ├── channel.cc                   # Channel setup and management
│   ├── transport.cc                 # Transport abstraction
│   │
│   ├── transport/                   # Transport implementations
│   │   ├── p2p.cc                  # GPU-to-GPU peer-to-peer (NVLink/PCIe)
│   │   ├── shm.cc                  # Shared memory (intra-node)
│   │   ├── net.cc                  # Network (inter-node)
│   │   ├── net_ib.cc               # InfiniBand transport
│   │   ├── net_socket.cc           # Socket transport
│   │   ├── nvls.cc                 # NVLink Sharp (NVLS)
│   │   └── coll_net.cc             # Collective network offload
│   │
│   ├── graph/                       # Topology and algorithm selection
│   │   ├── topo.cc                 # System topology detection
│   │   ├── search.cc               # Path finding algorithms
│   │   ├── rings.cc                # Ring algorithm topology
│   │   ├── trees.cc                # Tree algorithm topology
│   │   ├── tuning.cc               # Performance tuning tables
│   │   └── xml.cc                  # XML topology configuration
│   │
│   ├── device/                      # CUDA device-side code
│   │   ├── all_reduce.h            # AllReduce device templates
│   │   ├── all_gather.h            # AllGather device templates
│   │   ├── reduce_scatter.h        # ReduceScatter device templates
│   │   ├── broadcast.h             # Broadcast device templates
│   │   ├── sendrecv.h              # Point-to-point device code
│   │   ├── primitives.h            # Core device primitives
│   │   ├── prims_simple.h          # SIMPLE protocol primitives
│   │   ├── prims_ll.h              # Low-Latency protocol primitives
│   │   ├── prims_ll128.h           # LL128 protocol primitives
│   │   ├── common.h                # Device common utilities
│   │   └── reduce_kernel.h         # Reduction operations
│   │
│   ├── register/                    # Memory registration
│   │   ├── register.cc             # User buffer registration (ncclCommRegister)
│   │   ├── coll_reg.cc             # Collective buffer registration
│   │   └── sendrecv_reg.cc         # P2P buffer registration
│   │
│   ├── nccl_device/                 # Device API for user kernels
│   │   ├── core.cc                 # Device API core
│   │   ├── lsa_barrier.cc          # LSA (Load-Store-Atomic) barriers
│   │   ├── gin_barrier.cc          # GIN (GPU-initiated) barriers
│   │   └── ll_a2a.cc               # Low-latency AlltoAll
│   │
│   └── include/                     # Internal headers
│       ├── comm.h                  # ncclComm structure definition
│       ├── device.h                # Device structures and constants
│       ├── channel.h               # Channel structures
│       ├── proxy.h                 # Proxy structures
│       ├── transport.h             # Transport interface
│       ├── enqueue.h               # Enqueue interface
│       └── nccl_device/            # Device API headers
│           ├── gin.h               # GIN interface
│           └── impl/               # Device API implementations
│
└── examples/
    ├── 04_user_buffer_registration/ # Buffer registration examples
    ├── 05_symmetric_memory/         # Symmetric memory windows
    └── 06_device_api/               # Device API examples
```

## Architectural Layers

### Layer 1: Public Host API

The entry point is [nccl.h.in](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/nccl.h.in), which defines the public API. Users interact with NCCL through this interface.

**Key data types:**
- `ncclComm_t`: Opaque communicator handle (line 34)
- `ncclUniqueId`: Bootstrap identifier for communicator creation (line 39)
- `ncclWindow_t`: Registered memory window handle (line 35)

**Core operations** (all in [collectives.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/collectives.cc)):
- `ncclAllReduce()` (line 109): Reduce and broadcast result
- `ncclBroadcast()` (line 122): One-to-all distribution
- `ncclReduce()` (line 155): Many-to-one reduction
- `ncclAllGather()` (line 82): Gather from all ranks
- `ncclReduceScatter()` (line 168): Reduce and scatter result
- `ncclSend()`/`ncclRecv()` (lines 194/207): Point-to-point operations

Each collective API call constructs an `ncclInfo` structure and passes it to `ncclEnqueueCheck()` in [enqueue.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/enqueue.cc).

**Communicator lifecycle:**
1. `ncclGetUniqueId()` ([init.cc:158](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/init.cc#L158)): Generate bootstrap ID
2. `ncclCommInitRank()` ([init.cc:171](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/init.cc#L171)): Create communicator with rank
3. Use communicator for operations
4. `ncclCommFinalize()` ([nccl.h.in:188](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/nccl.h.in#L188)): Flush operations
5. `ncclCommDestroy()` ([nccl.h.in:192](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/nccl.h.in#L192)): Free resources

### Layer 2: Operation Enqueueing

When a user calls a collective operation, the flow goes through [enqueue.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/enqueue.cc). This is where NCCL transforms user requests into executable work for the GPU.

**The enqueueing pipeline:**

1. **Validation and Setup** (`ncclEnqueueCheck()`): Validates arguments, handles group semantics
2. **Work Creation**: Creates `ncclTaskColl` or `ncclTaskP2p` structures representing the operation
3. **Planning** ([enqueue.cc:182](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/enqueue.cc#L182)): `finishPlan()` creates a `ncclKernelPlan`
4. **Work Batching** ([enqueue.cc:110](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/enqueue.cc#L110)): `addWorkBatchToPlan()` groups operations
5. **Kernel Launch** ([enqueue.cc:28](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/enqueue.cc#L28)): `ncclInitKernelsForDevice()` prepares kernels

**Key structures** (from [comm.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/comm.h)):
- `ncclTaskColl` (line 191): Describes a collective operation (func, buffers, count, datatype, algorithm)
- `ncclKernelPlan` (line 252): Execution plan for one or more operations
- `ncclDevWorkBatch` (line 174): Batch of device work items

**Algorithm and protocol selection** is performed in the enqueue layer, choosing:
- **Algorithm**: Ring, Tree, CollNetDirect, CollNetChain, NVLS, NVLSTree, PAT
- **Protocol**: LL (Low-Latency), LL128, or SIMPLE

The tuning tables in [graph/tuning.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/graph/tuning.cc) guide these choices based on message size and topology.

### Layer 3: Transport Layer

The transport layer abstracts different communication mechanisms. The key abstraction is `ncclTransport` defined in [transport.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/transport.h#L118).

**Available transports:**
1. **P2P** ([transport/p2p.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/transport/p2p.cc)): Direct GPU-to-GPU via NVLink or PCIe
2. **SHM** ([transport/shm.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/transport/shm.cc)): Shared memory for intra-node
3. **NET** ([transport/net.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/transport/net.cc)): Network for inter-node (IB/Ethernet)
4. **COLLNET** ([transport/coll_net.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/transport/coll_net.cc)): Collective network offload
5. **NVLS** ([transport/nvls.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/transport/nvls.cc)): NVLink Sharp for intra-node

**Transport interface** ([transport.h:118](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/transport.h#L118)):
```c
struct ncclTransport {
  const char name[8];
  ncclResult_t (*canConnect)(int*, struct ncclComm*, ...);  // Can this transport connect these peers?
  struct ncclTransportComm send;  // Send-side operations
  struct ncclTransportComm recv;  // Receive-side operations
};
```

**Transport selection** happens during initialization:
1. **Topology detection** ([graph/topo.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/graph/topo.cc)): Discovers GPUs, NICs, PCIe topology, NVLink connectivity
2. **Path finding** ([graph/search.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/graph/search.cc)): Finds optimal paths between ranks
3. **Channel setup** ([channel.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/channel.cc)): Allocates channels and assigns transports

**Channels** are independent communication paths. NCCL typically uses multiple channels (default 4-32 depending on GPU) to maximize bandwidth. Each channel can use different transports for different peers.

### Layer 4: Device Primitives and Protocols

The device layer implements the actual GPU kernels that move data. This code lives in [src/device/](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/).

**Collective implementations:**
- [all_reduce.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/all_reduce.h): AllReduce implementation
- [all_gather.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/all_gather.h): AllGather implementation
- [reduce_scatter.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/reduce_scatter.h): ReduceScatter implementation
- [broadcast.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/broadcast.h): Broadcast implementation
- [sendrecv.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/sendrecv.h): Point-to-point operations

**Protocol primitives:**
Each protocol has its own primitive implementation:
- **SIMPLE** ([prims_simple.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/prims_simple.h)): Bulk data transfer, highest bandwidth
- **LL** ([prims_ll.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/prims_ll.h)): Low-latency with 8-byte flags, for small messages
- **LL128** ([prims_ll128.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/prims_ll128.h)): 128-byte granularity, balances latency and bandwidth

**Key device structures** (from [device.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/device.h)):
- `ncclDevKernelArgs` (passed to kernels): Contains channel mask, work storage type, device communicator pointer
- `ncclConnInfo` (line 128): Connection information for send/recv (buffers, head/tail pointers, step size)
- `ncclShmemData` ([device/common.h:42](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/common.h#L42)): Shared memory workspace

**The device-side execution model:**
1. Kernel launches with thread blocks assigned to channels
2. Each block loads its work batch from kernel args into shared memory ([common.h:130](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/common.h#L130))
3. Threads cooperate using primitives (Send, Recv, DirectSend, DirectRecv, Reduce)
4. Data flows through ring or tree patterns using connector buffers
5. Synchronization via head/tail pointers in `ncclConnInfo`

### Layer 5: Memory Registration

NCCL supports user buffer registration to optimize repeated operations on the same buffers. This is critical for zero-copy operation and enabling advanced features.

**Registration types:**

1. **User Buffer Registration** ([register/register.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/register/register.cc)):
   - API: `ncclCommRegister()` / `ncclCommDeregister()` ([nccl.h.in:265](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/nccl.h.in#L265))
   - Purpose: Register CUDA buffers for zero-copy operation
   - Example: [04_user_buffer_registration](file:///home/jeromeku/torchcomms/thirdparty/nccl/examples/04_user_buffer_registration/)

2. **Symmetric Memory Windows** ([register/coll_reg.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/register/coll_reg.cc)):
   - API: `ncclCommWindowRegister()` / `ncclCommWindowDeregister()` ([nccl.h.in:273](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/nccl.h.in#L273))
   - Purpose: Enable symmetric collective operations with consistent memory layouts
   - Requirement: Buffers allocated via CUDA VMM (`ncclMemAlloc()`)
   - Flag: `NCCL_WIN_COLL_SYMMETRIC` (line 59)
   - Example: [05_symmetric_memory](file:///home/jeromeku/torchcomms/thirdparty/nccl/examples/05_symmetric_memory/)

**Registration lifecycle:**
1. Allocate buffer (via `ncclMemAlloc()` for symmetric, or `cudaMalloc()` for regular)
2. Register with `ncclCommRegister()` or `ncclCommWindowRegister()`
3. Use in collectives (same API calls, but NCCL optimizes registered buffers)
4. Deregister when done
5. Free buffer

**Transport-specific registration:**
- **IPC registration** ([transport.h:159](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/transport.h#L159)): `ncclRegisterP2pIpcBuffer()` for peer GPU access
- **Network registration** ([transport.h:160](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/transport.h#L160)): `ncclRegisterP2pNetBuffer()` for RDMA
- **NVLS registration** ([transport.h:134](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/transport.h#L134)): `ncclNvlsGraphRegisterBuffer()` for NVLink Sharp

### Layer 6: Proxy Threads

For network operations (NET, COLLNET), NCCL uses proxy threads to handle asynchronous progress. The proxy layer lives in [proxy.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/proxy.cc).

**Why proxies?** GPU kernels cannot directly call network APIs (InfiniBand verbs, socket send/recv). Proxies provide:
- Asynchronous network progress without blocking the GPU
- CPU-side handling of network completion events
- Buffer management for network operations

**Proxy architecture:**
1. **Main proxy thread** per communicator (spawned during init)
2. **Proxy operations** (`ncclProxyOp`) queued from enqueue layer ([proxy.h:55](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/proxy.h#L55))
3. **Proxy progress** functions handle transport-specific operations
4. **Synchronization** via FIFO between GPU and proxy ([device.h:141](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/device.h#L141))

**Key structures** ([proxy.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/proxy.h)):
- `ncclProxyOp` (line 55): Describes one proxy operation (send/recv, size, channel, pattern)
- `ncclProxyArgs` (line 164): Aggregated proxy arguments for execution
- `ncclProxyState`: Proxy thread state and queues

**Proxy operation flow:**
1. Enqueue layer adds proxy ops to plan ([enqueue.cc:99](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/enqueue.cc#L99))
2. Proxy thread polls for new operations
3. Executes transport-specific progress function (e.g., IB post_send/poll_cq)
4. Updates completion via shared FIFO
5. GPU kernel polls FIFO to know when network operation completes

### Layer 7: Device API and LSA Barriers

NCCL provides a device API that allows users to call NCCL operations from within their own CUDA kernels. This enables fusing communication with computation. The implementation is in [src/nccl_device/](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/nccl_device/).

**Device API components:**

1. **Device Communicator** ([nccl_device/core.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/nccl_device/core.cc)):
   - Host creates: `ncclDevCommCreate()` (requires `ncclDevCommRequirements` specifying LSA barrier count)
   - Device uses: `ncclDevComm` structure for device-side operations
   - Example: [06_device_api](file:///home/jeromeku/torchcomms/thirdparty/nccl/examples/06_device_api/)

2. **LSA Barriers** ([nccl_device/lsa_barrier.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/nccl_device/lsa_barrier.cc)):
   - **LSA** = Load-Store-Atomic operations
   - Purpose: Synchronize CUDA thread blocks across GPUs
   - Used for device API to coordinate collective operations
   - Enables user kernels to perform cross-GPU barriers

3. **Symmetric Memory Access** (requires `ncclCommWindowRegister()` with `NCCL_WIN_COLL_SYMMETRIC`):
   - Provides LSA-accessible pointers to peer GPU memory
   - Allows direct load/store to remote GPU memory from device code
   - Required for device API collectives

**Device API usage pattern:**
```c
// Host side:
ncclDevComm devComm;
ncclDevCommRequirements reqs = {};
reqs.lsaBarrierCount = NCCL_DEVICE_CTA_COUNT;
ncclDevCommCreate(comm, &reqs, &devComm);
ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC);

// Device kernel:
myKernel<<<grid, block>>>(devComm, win) {
  // Use LSA barrier to sync across GPUs
  ncclDeviceLsaBarrier(...);

  // Access peer memory via symmetric window
  // Perform custom computation + communication
}
```

**GIN (GPU-Initiated Networking)** ([nccl_device/gin_barrier.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/nccl_device/gin_barrier.cc)):
- Experimental feature for direct GPU-to-network communication
- Bypasses CPU proxy for certain network operations
- Headers in [include/gin/](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/gin/)

## Data Flow

Let's trace a typical `ncclAllReduce()` call through the system:

### 1. API Call
```c
ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream);
```
Enters [collectives.cc:109](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/collectives.cc#L109) → creates `ncclInfo` → calls `ncclEnqueueCheck()`

### 2. Enqueueing
[enqueue.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/enqueue.cc):
- Validates arguments and communicator state
- Creates `ncclTaskColl` structure with operation details
- If in a group, adds to group task list; otherwise proceeds immediately
- Determines algorithm (Ring/Tree/NVLS) and protocol (LL/LL128/SIMPLE) based on message size and topology
- Creates `ncclKernelPlan` with work batches

### 3. Kernel Launch
- Allocates work in device-visible memory or kernel args
- Copies work batches and tasks to GPU
- If network involved, creates proxy operations and queues them
- Launches CUDA kernel on user's stream with specialized function pointer

### 4. Device Execution
Kernel entry in [device/all_reduce.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/all_reduce.h):
- Each thread block handles one channel
- Loads work batch into shared memory ([device/common.h:130](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/common.h#L130))
- Threads coordinate via barriers ([device/common.h:78](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/common.h#L78))
- Executes ring or tree algorithm:
  - **Ring**: Data flows in a circle, each rank reduces with incoming data and forwards
  - **Tree**: Binary tree reduction up, then broadcast down
  - **NVLS**: Uses multicast memory for intra-node reduction

### 5. Data Movement (Ring Example)
Using primitives from [device/prims_simple.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/prims_simple.h):
- **Step 0**: Each rank sends its chunk to next neighbor, receives from previous
- **Steps 1 to N-2**: Reduce received chunk with local data, forward to next
- **Final steps**: Complete the ring, all ranks have full reduction

**Transport-specific paths:**
- **NVLink/PCIe** (P2P): Direct GPU load/store or DMA
- **Shared Memory** (SHM): Copy via `/dev/shm`
- **Network** (NET): Write to send buffer → Proxy posts network send → Remote proxy receives → Copy to receive buffer → GPU reads

### 6. Proxy Operations (if network involved)
[proxy.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/proxy.cc):
- Proxy thread polls for new `ncclProxyOp` structures
- Executes network operations (IB post_send, poll_cq, etc.)
- Updates FIFO to signal GPU when data arrives
- GPU kernel polls FIFO and proceeds when ready

### 7. Completion
- Kernel completes when all steps finish
- CUDA stream ordering ensures subsequent operations wait
- User can check completion via stream synchronization or events

## Key Abstractions

### Communicator (ncclComm_t)

The communicator is the central abstraction, defined in [include/comm.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/comm.h). It encapsulates all state for a group of cooperating ranks.

**Key fields:**
- `rank`, `nRanks`: This rank's ID and total ranks
- `cudaDev`: CUDA device this communicator is bound to
- `channels[]`: Array of `ncclChannel` structures (communication paths)
- `devComm`: Device-side communicator pointer
- `proxyState`: Proxy thread state for network operations
- `memPermanent`, `memScoped`: Memory allocators

**Shared resources** ([comm.h:118](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/comm.h#L118)):
- Communicators can share resources (channels, proxy threads) via split/shrink operations
- `splitShare` flag in `ncclConfig_t` controls sharing

### Channels and Connections

A **channel** ([include/comm.h:148](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/comm.h#L148)) is an independent communication path:
- Has its own ring and tree topology (`ring`, `tree` fields)
- Contains array of `ncclChannelPeer*` for peer connections
- Each peer connection has send and receive `ncclConnector`

A **connector** ([device.h:159](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/device.h#L159)) represents one directional connection:
- Points to transport-specific resources (`transportComm`, `transportResources`)
- Contains `ncclConnInfo` with buffer pointers and synchronization primitives
- Has `proxyConn` for proxy thread communication if needed

### Algorithms and Patterns

NCCL implements collectives using different **algorithms**:

1. **Ring** ([graph/rings.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/graph/rings.cc)):
   - Data split into chunks, circulates through ranks
   - Bandwidth-optimal: transfers data exactly once per link
   - Best for large messages
   - Pattern: Each rank sends to next, receives from previous

2. **Tree** ([graph/trees.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/graph/trees.cc)):
   - Binary tree reduction then broadcast
   - Latency-optimal: log(N) steps
   - Best for small messages or latency-sensitive operations
   - Pattern: Reduce up tree, broadcast down

3. **CollNet** (Direct and Chain):
   - Offload to network switch with collective support (e.g., SHARP)
   - Reduces number of network hops
   - Requires special hardware/network plugin

4. **NVLS** ([transport/nvls.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/transport/nvls.cc)):
   - NVLink Sharp for intra-node
   - Uses multicast memory for efficient reduction
   - Requires Hopper+ GPUs with NVLink support

**Patterns** ([proxy.h:21](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/proxy.h#L21)): Encode communication flow
- `ncclPatternRing`, `ncclPatternRingTwice`: Ring traversals
- `ncclPatternTreeUp`, `ncclPatternTreeDown`: Tree phases
- `ncclPatternPipelineFrom`, `ncclPatternPipelineTo`: Pipeline patterns
- `ncclPatternSend`, `ncclPatternRecv`: Point-to-point

### Protocols

Three **protocols** control how data is transferred:

1. **SIMPLE** (from [device/prims_simple.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/prims_simple.h)):
   - Bulk data transfer without per-element flags
   - Highest bandwidth, used for large messages
   - Synchronization via head/tail pointers in `ncclConnInfo`
   - Step size: Full buffer divided by NCCL_STEPS (typically 8)

2. **LL (Low-Latency)** (from [device/prims_ll.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/prims_ll.h)):
   - Each 8 bytes of data paired with 8 bytes of flags
   - Fine-grained synchronization, lower latency
   - Used for small messages (< 32KB typically)
   - Flag checking: `NCCL_LL_FLAG()` macro ([device.h:100](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/device.h#L100))

3. **LL128** (from [device/prims_ll128.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/prims_ll128.h)):
   - 128-byte granularity with flag in last 8 bytes
   - Balances bandwidth and latency
   - Used for medium messages
   - Constants: `NCCL_LL128_LINESIZE` (128), `NCCL_LL128_DATAELEMS` (15 of 16 elements)

**Protocol selection** happens in the enqueue layer based on message size and tuning tables.

### Device Communicator (ncclDevComm)

For the device API, NCCL provides `ncclDevComm` - a GPU-accessible communicator structure. Created via:
```c
ncclDevCommCreate(ncclComm_t hostComm, ncclDevCommRequirements* reqs, ncclDevComm* devComm);
```

**Requirements** (`ncclDevCommRequirements`):
- `lsaBarrierCount`: Number of LSA barriers needed (typically `NCCL_DEVICE_CTA_COUNT` from example)
- Specifies resources needed for device-side operations

**LSA Barriers** enable cross-GPU synchronization from device code:
- Implemented using remote atomic operations on peer GPU memory
- Requires symmetric memory windows for peer access
- Used to coordinate phases of device-initiated collectives

## Important Implementation Details

### Topology Detection and Path Selection

During initialization ([init.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/init.cc)), NCCL detects:
- GPU locations via CUDA and NVML APIs
- PCIe topology via `/sys/class/pci_device/`
- NVLink connectivity via NVML
- Network interfaces via system calls
- NUMA affinity

**Topology graph** ([graph/topo.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/graph/topo.cc)):
- Builds graph representation of system
- Nodes: GPUs, CPUs, NICs, switches
- Edges: Interconnects with bandwidth/latency estimates
- Can be overridden via XML file ([graph/xml.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/graph/xml.cc))

**Search algorithm** ([graph/search.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/graph/search.cc)):
- Finds best paths for each collective pattern
- Optimizes for bandwidth and latency
- Considers constraints (e.g., avoid PCIe switches when possible)
- Results in ring and tree assignments for each channel

### Bootstrap Protocol

Before the main communicator is set up, ranks use a **bootstrap protocol** ([bootstrap.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/bootstrap.cc)) for out-of-band communication:

1. Rank 0 calls `ncclGetUniqueId()` → creates listening socket
2. UniqueId (containing socket address) distributed to all ranks (via MPI, files, etc.)
3. All ranks connect to rank 0's bootstrap socket
4. Bootstrap used for initial AllGather of rank information (peer info, topology)
5. Once main channels established, bootstrap protocol no longer needed

**Bootstrap operations:**
- `bootstrapNetInit()` (line 93): Initialize network for bootstrap
- `bootstrapGetUniqueId()`: Create listening socket and return address
- Bootstrap AllGather: Exchange peer info across all ranks

### Group Semantics

NCCL's `ncclGroupStart()`/`ncclGroupEnd()` ([group.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/group.cc)) allows batching operations:

```c
ncclGroupStart();
for (int i = 0; i < nGPUs; i++) {
  ncclAllReduce(..., comms[i], streams[i]);
}
ncclGroupEnd();  // All enqueued operations launch together
```

**Benefits:**
- Enables multi-GPU programming from single thread
- Allows operation fusion and optimization across GPUs
- Required for deadlock-free send/recv pairs
- Can improve performance by batching kernel launches

**Implementation:**
- Group calls don't immediately enqueue kernels
- Operations accumulated in per-thread state
- `ncclGroupEnd()` processes all accumulated operations atomically
- Simulator mode: `ncclGroupSimulateEnd()` estimates time without execution

### Memory Registration Deep Dive

Memory registration optimizes repeated operations on the same buffers:

**Regular registration** (`ncclCommRegister()`):
1. Pins memory if needed (prevents paging)
2. Registers with transports (IB, GPU P2P, etc.)
3. Creates handle for later use
4. NCCL detects registered buffers automatically in subsequent collectives

**Symmetric registration** (`ncclCommWindowRegister()` with `NCCL_WIN_COLL_SYMMETRIC`):
1. Requires buffers allocated via `ncclMemAlloc()` (uses CUDA VMM API)
2. Creates symmetric mapping across all ranks
3. Each rank maps peer memory at same virtual address
4. Enables LSA (load-store-atomic) access from device code
5. Required for device API operations

**Registration types** by transport:
- **IPC** (GPU P2P): `cudaIpcGetMemHandle()`, map in remote process
- **Network**: Plugin-specific registration (IB: `ibv_reg_mr()`)
- **NVLS**: Multicast handle creation and binding

**Deregistration** must happen before buffer free to avoid leaks.

### Kernel Specialization

NCCL generates specialized kernels for different configurations:
- Function/algorithm/protocol combinations
- Compile-time specialization via templates in device headers
- Runtime kernel selection based on `devFuncId` computed during enqueue
- Kernel function table: `ncclDevFuncTable[]` (from [device/common.h:27](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/device/common.h#L27))

**Symmetric kernels** ([sym_kernels.cc](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/sym_kernels.cc)):
- Special kernels for symmetric memory windows
- Can use optimized access patterns
- Reduced overhead from registration checks

### Profiler Integration

NCCL supports external profiler plugins ([include/profiler.h](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/include/profiler.h)):
- Hooks at enqueue, launch, and completion
- Per-operation event handles
- Proxy profiling for network operations
- Integrates with NVTX for NSight Systems

### Tuner Plugin

External tuning plugins ([plugin/tuner/](file:///home/jeromeku/torchcomms/thirdparty/nccl/src/plugin/tuner/)):
- Override default algorithm/protocol selection
- Implement `ncclTuner_t` interface
- Can customize based on message size, topology, history

## Summary

NCCL's architecture is structured in clear layers:

1. **API Layer**: User-facing C API with communicators and collectives
2. **Enqueue Layer**: Plans operations, selects algorithms/protocols, creates work batches
3. **Transport Layer**: Abstraction over P2P/SHM/NET/COLLNET/NVLS
4. **Device Layer**: CUDA kernels implementing ring/tree algorithms with protocol-specific primitives
5. **Proxy Layer**: CPU threads handling asynchronous network progress
6. **Bootstrap**: Out-of-band setup protocol for initialization

**Data flows** from user API call → enqueue/planning → CUDA kernel execution → transport-specific data movement → completion.

**Key innovations:**
- Multiple channels for bandwidth aggregation
- Algorithm/protocol co-optimization
- Topology-aware path selection
- Symmetric memory for device API
- Zero-copy with registration
- Asynchronous proxy threads for networks

The codebase is highly optimized for performance, with careful attention to:
- Minimizing synchronization overhead
- Maximizing bandwidth utilization
- Reducing latency through protocol selection
- Enabling advanced features (device API, symmetric memory)

For more details, see:
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/)
- [NCCL API Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html)
- [Examples](file:///home/jeromeku/torchcomms/thirdparty/nccl/examples/)
