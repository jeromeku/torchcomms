# NCCL Architecture and Implementation Documentation

Comprehensive documentation of NVIDIA Collective Communications Library (NCCL) internals, created for developers who want to understand how NCCL works from user-facing APIs down to CUDA kernels.

## Contents

### 1. [ARCHITECTURE.md](ARCHITECTURE.md)
**High-level system architecture document** following the style from [matklad.github.io](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html)

- Bird's eye view of NCCL's architecture
- Directory structure and code organization
- Seven architectural layers explained in detail
- Data flow from API calls to GPU operations
- Key abstractions: communicators, channels, protocols, algorithms
- Implementation details: topology, bootstrap, memory registration

**Size**: 605 lines | 34 KB

**Start here** to get oriented with NCCL's overall architecture.

---

### 2. [trace_04_user_buffer_registration.md](trace_04_user_buffer_registration.md)
**Complete execution trace for User Buffer Registration**

Demonstrates how `ncclCommRegister` pre-registers buffers for optimized communication performance.

**Topics covered:**
- `ncclMemAlloc`: CUDA VMM allocation with FABRIC/RDMA handles
- `ncclCommRegister`: Registration cache and handle creation
- `ncclAllReduce`: How registered buffers are detected and used
- `ncclCommDeregister`: Cleanup and transport deregistration
- Performance benefits of buffer registration (10-30% for small messages)

**Key insights:**
- Two-phase registration: cache entry creation + lazy transport registration
- Registration cache with page-aligned ranges and refcounting
- Specialized device work types for registered buffers

**Size**: 1,240 lines | 46 KB

---

### 3. [trace_05_symmetric_memory.md](trace_05_symmetric_memory.md)
**Complete execution trace for Symmetric Memory Windows**

Demonstrates `ncclCommWindowRegister` with `NCCL_WIN_COLL_SYMMETRIC` flag for optimized collective operations.

**Topics covered:**
- Symmetric memory concept: unified flat VA space across GPUs
- `ncclCommWindowRegister`: Window creation and collective exchange
- Bootstrap protocol for memory handle exchange
- CUDA VMM mapping to create symmetric address space
- Specialized symmetric kernels with direct peer access
- NVLS multicast support on Hopper+ GPUs

**Key insights:**
- "Big VA space" strategy: predictable offsets for peer access
- LSA (Local Symmetric Access) teams
- Direct peer reads/writes via `peerPtr()` calculations
- Up to 1000 GB/s bandwidth with multicast

**Size**: 1,437 lines | 59 KB

---

### 4. [trace_06_device_api.md](trace_06_device_api.md)
**Complete execution trace for Device API**

Demonstrates how GPU kernels can perform collective operations directly using NCCL's Device API.

**Topics covered:**
- `ncclDevCommCreate`: Device communicator creation with LSA barriers
- LSA barrier synchronization across GPUs
- `ncclGetLsaPointer`: Direct peer memory access
- Device kernel implementation with cross-GPU coordination
- Memory ordering semantics (relaxed, acquire, release)

**Key insights:**
- Kernel-initiated communication without CPU intervention
- Epoch-based LSA barrier protocol using atomics
- One barrier per CTA for proper coordination
- Lower latency (~10 μs) for small operations vs Host API (~17 μs)
- Best for compute-communication fusion

**Size**: 2,643 lines | 91 KB

---

## Documentation Style

All traces follow a **"literate code"** approach:

✅ **Line-by-line walkthrough** of the complete call stack
✅ **Annotated code snippets** from actual source files
✅ **Clickable file references** (VSCode markdown links)
✅ **Source file + line number spans** for every function
✅ **Data structure definitions** with field explanations
✅ **Sequence diagrams** using ASCII art
✅ **Performance analysis** with concrete measurements
✅ **Comparison tables** highlighting differences

## How to Use This Documentation

### For First-Time Readers
1. Start with [ARCHITECTURE.md](ARCHITECTURE.md) for the big picture
2. Read [trace_04_user_buffer_registration.md](trace_04_user_buffer_registration.md) to understand basic flow
3. Progress to [trace_05_symmetric_memory.md](trace_05_symmetric_memory.md) for advanced optimizations
4. Finally [trace_06_device_api.md](trace_06_device_api.md) for device-side programming

### For Specific Topics

**Memory allocation:**
- ARCHITECTURE.md → "Memory Registration" section
- trace_04 → "ncclMemAlloc" section

**Collective operations:**
- ARCHITECTURE.md → "Data Flow" section
- trace_04 → "ncclAllReduce" complete call path

**Symmetric memory and peer access:**
- trace_05 → Complete walkthrough
- ARCHITECTURE.md → "Layer 5: Memory Registration"

**Device-side communication:**
- trace_06 → Complete walkthrough
- ARCHITECTURE.md → "Layer 7: Device API"

**LSA barriers:**
- trace_06 → "LSA Barrier Deep Dive"

**Kernel implementation:**
- trace_04 → Device kernel selection and templates
- trace_05 → Symmetric kernel primitives
- trace_06 → Custom device kernels

## Source Code References

All file references use clickable markdown links relative to the repository root:

```markdown
[filename.cc](../../../thirdparty/nccl/src/filename.cc)
[filename.cc:123](../../../thirdparty/nccl/src/filename.cc#L123)
```

Click on any link to jump directly to the source code in your editor.

## Examples Traced

The three examples progressively demonstrate advanced NCCL features:

| Example | Feature | Complexity | Use Case |
|---------|---------|------------|----------|
| [04_user_buffer_registration](../../../thirdparty/nccl/examples/04_user_buffer_registration) | Buffer pre-registration | Low | Optimize repeated collectives on same buffers |
| [05_symmetric_memory](../../../thirdparty/nccl/examples/05_symmetric_memory) | Symmetric VA windows | Medium | Intra-node collectives with direct peer access |
| [06_device_api](../../../thirdparty/nccl/examples/06_device_api) | Device-initiated communication | High | Fused compute-communication kernels |

## Related Resources

**NCCL Source**: [thirdparty/nccl/](../../../thirdparty/nccl/)
**NCCL Examples**: [thirdparty/nccl/examples/](../../../thirdparty/nccl/examples/)
**Official NCCL Docs**: https://docs.nvidia.com/deeplearning/nccl/
**CUDA VMM API**: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
**NVLS/Multicast**: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html

---

**Total Documentation**: 5,925 lines | 230 KB

Created with the goal of providing **complete transparency** into NCCL's implementation from user API down to GPU execution.
