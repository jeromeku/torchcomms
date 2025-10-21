# NCCLX vs NCCL — Architecture, Diffs, and APIs

Table of Contents
- [At A Glance](#at-a-glance)
- [Directory Layout](#directory-layout)
- [Forwarding Headers](#forwarding-headers)
- [Public Header Extensions](#public-header)
- [CTRAN API](#ctran-api)
- [CTRAN Integration](#ctran-integration)
- [New Collectives](#new-collectives)
- [Debug and Cvars](#debug-cvars)
- [Build and Link](#build-link)
- [AllToAllv Dataflow](#alltoallv-dataflow)
- [New APIs and Flags](#new-apis-flags)
- [Selected Diffs](#selected-diffs)
- [Feature Comparison](#feature-comparison)
- [Where Next](#where-next)
- [Appendix: Invocation Map](#appendix-invocation)

- Scope: Compare `comms/ncclx` against upstream `thirdparty/nccl`.
- Artifacts generated under `codex/`:
  - `codex/nccl_vs_ncclx_v2_27.diff` — full directory diff of `thirdparty/nccl` vs `comms/ncclx/v2_27`.
  - `codex/diff_quick_v2_27.txt` — quick summary (which files differ / are added).
  - `codex/files_common_rel.txt` — paths common to both trees (relative).
  - `codex/files_only_upstream_vs_v2_27.txt` — only in upstream NCCL.
  - `codex/files_only_ncclx_v2_27.txt` — only in NCCLX v2_27.
  - `codex/sample_api_*` — quick API symbol samples.

<a id="at-a-glance"></a>
## At A Glance

- NCCLX embeds upstream NCCL sources under `comms/ncclx/v2_27` and layers additional “meta” functionality: CTRAN transport, persistent collectives, all-to-all(v), colltrace, comms monitor, hints/cvars, and Python bindings.
- NCCLX exports forwarding headers in `comms/ncclx/headers/` to avoid conflicts and support header namespacing.
- API surface extends beyond vanilla NCCL:
  - New C++-level features (persistent collectives, AllToAll/AllToAllv), `NCCL_COMM_WORLD`, RMA window support flags, and runtime “hints”.
  - New CTRAN external API (`comms/ctran/CtranEx.h`) + NCCL adapter (`CtranExComm`).
  - Python module `ncclx_trainer_context` for training hints.
- Build/packaging diverges to include `version.script` for symbol visibility and to link in META/CTRAN and utils code across the repo.

<a id="directory-layout"></a>
## Directory Layout

- Upstream NCCL: `thirdparty/nccl/`
  - Core sources under `src/`, plus `examples/`, `ext-*`, `makefiles/`, `pkg/`.
- NCCLX: `comms/ncclx/`
  - `v2_27/` mirrors upstream `src/` layout but adds `meta/` (extensions) and build integration for this repo.
  - `headers/` contains forwarding headers to manage include order and namespacing.

Files unique to each side and common paths are listed here:
- `codex/files_only_upstream_vs_v2_27.txt`
- `codex/files_only_ncclx_v2_27.txt`
- `codex/files_common_rel.txt`

See quick summary of differing/added files in `codex/diff_quick_v2_27.txt` and the full diff in `codex/nccl_vs_ncclx_v2_27.diff`.

<a id="forwarding-headers"></a>
## Forwarding Headers (Namespacing)

- NCCLX adds forwarding public headers to select the correct `nccl.h` under different build layouts.

  [comms/ncclx/headers/nccl.h:1](../comms/ncclx/headers/nccl.h#L1)
  #include "comms/ncclx/headers/helpers.h"
  #ifdef HEADER_NAMESPACE
  #include NAMESPACED_HEADER_PATH(HEADER_NAMESPACE, nccl.h)
  #else
  #include HEADER_PATH(nccl.h)
  #endif

  [comms/ncclx/headers/helpers.h:5](../comms/ncclx/headers/helpers.h#L5)
  #define NAMESPACED_HEADER_PATH(header_namespace, header) <header_namespace/header>
  #define HEADER_PATH(header) <header>

- Also forwards `CtranExComm.h`:

  [comms/ncclx/headers/CtranExComm.h:6](../comms/ncclx/headers/CtranExComm.h#L6)
  #include NAMESPACED_HEADER_PATH(HEADER_NAMESPACE, CtranExComm.h)

This ensures NCCLX headers win when both NCCL and NCCLX are present.

<a id="public-header"></a>
## Public Header (nccl.h) — NCCLX Extensions

- NCCLX generates a public header from `comms/ncclx/v2_27/src/nccl.h.in` with additional macros and APIs:

  [comms/ncclx/v2_27/src/nccl.h.in:19](../comms/ncclx/v2_27/src/nccl.h.in#L19)
  #define NCCL_MAJOR 2
  #define NCCL_MINOR 27
  #define NCCL_PATCH 7
  #define NCCL_SUFFIX "x-${nccl:DevSignature}"
  #define IS_NCCLX

- New capability flags (not present in upstream header):

  [comms/ncclx/v2_27/src/nccl.h.in:37](../comms/ncclx/v2_27/src/nccl.h.in#L37)
  #define NCCL_COMM_DESCRIPTION
  #define NCCL_COMM_SPLIT_GROUP_RANKS_SUPPORTED
  #define NCCL_ALLTOALLV_SUPPORTED
  #define NCCL_ALLTOALL_SUPPORTED
  #define NCCL_PERSISTENT_COLL_SUPPORTED
  #define NCCL_COMM_ALGO_CONTROL_SUPPORTED
  #define NCCL_COMM_DUMP
  #define NCCL_COMM_GET_UNIQUE_HASH
  #define NCCL_COLLTRACE_CUDA_GRAPH_COMPATIBLE

- RMA and COMM_WORLD support:

  [comms/ncclx/v2_27/src/nccl.h.in:52](../comms/ncclx/v2_27/src/nccl.h.in#L52)
  extern ncclComm_t ncclCommWorld;  // exposed as NCCL_COMM_WORLD
  #define NCCL_COMM_WORLD ncclCommWorld

  [comms/ncclx/v2_27/src/nccl.h.in:58](../comms/ncclx/v2_27/src/nccl.h.in#L58)
  #define NCCL_RMA_SUPPORTED
  typedef struct ncclWin* ncclWin_t;

- New NCCLX APIs (C++ block at tail of the header):

  [comms/ncclx/v2_27/src/nccl.h.in:13](../comms/ncclx/v2_27/src/nccl.h.in#L13)
  ncclResult_t AllToAllInit(void* recvbuff, size_t maxRecvCount, const Hints& hints,
                            ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,
                            void*& request);

  [comms/ncclx/v2_27/src/nccl.h.in:26](../comms/ncclx/v2_27/src/nccl.h.in#L26)
  ncclResult_t AllToAllExec(const void* sendbuff, size_t count, void* request);

  [comms/ncclx/v2_27/src/nccl.h.in:34](../comms/ncclx/v2_27/src/nccl.h.in#L34)
  ncclResult_t pFree(void* request);

  [comms/ncclx/v2_27/src/nccl.h.in:36](../comms/ncclx/v2_27/src/nccl.h.in#L36)
  std::shared_ptr<const std::unordered_map<std::string, std::string>> getNcclxInfo();

  [comms/ncclx/v2_27/src/nccl.h.in:41](../comms/ncclx/v2_27/src/nccl.h.in#L41)
  ncclResult_t setGlobalHint(std::string key, std::string val);

Contrast with upstream header, which lacks these flags and functions:

  [thirdparty/nccl/src/nccl.h.in:24](../thirdparty/nccl/src/nccl.h.in#L24)
  #define NCCL_VERSION_CODE ${nccl:Version}
  // ... no IS_NCCLX, no NCCL_*_SUPPORTED, no AllToAll/Hint APIs

<a id="ctran-api"></a>
## New External API: CTRAN (Explicit Transport)

- NCCLX ships a transport extension API that is not part of upstream NCCL:

  [comms/ctran/CtranEx.h:51](../comms/ctran/CtranEx.h#L51)
  class __attribute__((visibility("default"))) CtranEx {
    CtranEx(int rank, int cudaDevice, const CtranExHostInfo& hostInfo,
            const std::vector<CtranExBackend> backends, const std::string& desc);
    bool isInitialized();
    commResult_t regMem(const void* ptr, size_t size, void** regHdl);
    commResult_t iput(const void* localBuf, size_t len, void* localRegHdl,
                      int peerRank, void* peerRemoteBuf, uint32_t peerRemoteKey,
                      bool notify, CtranExRequest** req);
    // ... more control plane and flush/notify APIs
  };

- An adapter that wraps `ncclComm_t` to use CTRAN from NCCL side:

  [comms/ncclx/v2_27/meta/wrapper/CtranExComm.h:20](../comms/ncclx/v2_27/meta/wrapper/CtranExComm.h#L20)
  class __attribute__((visibility("default"))) CtranExComm {
    CtranExComm(const ncclComm_t comm, const std::string& commDesc);
    ncclResult_t regMem(const void* ptr, size_t size, void** segHdl, bool forceRegister=false);
    ncclResult_t broadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root, CtranExRequest** req);
  };

These APIs are entirely new relative to upstream NCCL.

<a id="ctran-integration"></a>
## CTRAN Integration in NCCLX Core

- NCCLX integrates CTRAN into the communicator initialization path and runtime:

  [comms/ncclx/v2_27/src/init.cc:1627](../comms/ncclx/v2_27/src/init.cc#L1627)
  /* NCCLX - Specific Initialization */
  NCCLCHECKGOTO(meta::comms::ncclx::newCollTraceInit(comm), res, fail);
  NCCLCHECKGOTO(metaCommToNccl(setCtranCommBase(comm)), res, fail);
  comm->ctranComm_->bootstrap_ = std::make_unique<ncclx::BaselineBootstrap>(comm);
  comm->ctranComm_->statex_ = ncclx::createCommStateXFromNcclComm(comm);
  NCCLCHECKGOTO(metaCommToNccl(ncclx::transport::tranportProxyInit(comm, job->parent)), res, fail);
  if (comm->useCtran_) {
    NCCLCHECK(ncclx::initCtranCommStatexFromNcclComm(comm, comm->ctranComm_.get()));
    comm->ctranComm_->colltraceNew_ = comm->newCollTrace;
    NCCLCHECKGOTO(metaCommToNccl(ctranInit(comm->ctranComm_.get())), res, fail);
  }

- Global communicator symbol for convenience (not upstream):

  [comms/ncclx/v2_27/src/init.cc:66](../comms/ncclx/v2_27/src/init.cc#L66)
  ncclComm_t ncclCommWorld __attribute__ ((visibility("default")));

<a id="new-collectives"></a>
## New Collectives and Algorithm Control

- AllToAllv (new) — with CTRAN fallback and dynamic variants:

  [comms/ncclx/v2_27/src/collectives.cc:350](../comms/ncclx/v2_27/src/collectives.cc#L350)
  NCCL_API(ncclResult_t, ncclAllToAllv, const void* sendbuff, const size_t sendcounts[],
           const size_t sdispls[], void* recvbuff, const size_t recvcounts[],
           const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

  [comms/ncclx/v2_27/src/collectives.cc:397](../comms/ncclx/v2_27/src/collectives.cc#L397)
  if ((NCCL_ALLTOALLV_ALGO == NCCL_ALLTOALLV_ALGO::ctran) && ctranAllToAllvSupport(comm->ctranComm_.get())) {
    return metaCommToNccl(ctranAllToAllv(...));
  }
  // fallback to baseline send/recv based alltoallv

- Persistent collectives (request-based) appear in headers and meta layer (`meta/collectives/pCollectives.cc`).

<a id="debug-cvars"></a>
## Debug/Logging and Config Vars

- NCCLX replaces upstream param/env plumbing with a central cvars system and extends debug logging:

  [comms/ncclx/v2_27/src/debug.cc:37](../comms/ncclx/v2_27/src/debug.cc#L37)
  int tempNcclDebugLevel = static_cast<int>(meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG));
  ncclDebugMask = meta::comms::logger::parseDebugSubsysMask(NCCL_DEBUG_SUBSYS.c_str());

  [comms/utils/cvars/nccl_cvars.h:14](../comms/utils/cvars/nccl_cvars.h#L14)
  extern bool CUDA_LAUNCH_BLOCKING;  // generated; many NCCL_* toggles declared here

- Upstream param header includes env loaders and `NCCL_PARAM` macro. NCCLX simplifies the header to initialization stubs and logger init:

  [thirdparty/nccl/src/include/param.h:12](../thirdparty/nccl/src/include/param.h#L12)
  const char* userHomeDir();
  void setEnvFile(const char* fileName);
  void initEnv();
  const char *ncclGetEnv(const char *name);
  #define NCCL_PARAM(name, env, deftVal) ...

  [comms/ncclx/v2_27/src/include/param.h:12](../comms/ncclx/v2_27/src/include/param.h#L12)
  void initEnv();
  void initNcclLogger();

<a id="build-link"></a>
## Build/Link Differences

- NCCLX `src/Makefile` pulls in META, CTRAN, and utils from the top-level repo and uses a `version.script` to control symbol exports:

  [comms/ncclx/v2_27/src/Makefile:24](../comms/ncclx/v2_27/src/Makefile#L24)
  INCEXPORTS := nccl.h CtranEx.h CtranExComm.h

  [comms/ncclx/v2_27/src/Makefile:37](../comms/ncclx/v2_27/src/Makefile#L37)
  LIBSRCFILES += $(wildcard ${NCCLDIR}/meta/*.cc ... meta/wrapper/*.cc)

  [comms/ncclx/v2_27/src/Makefile:210](../comms/ncclx/v2_27/src/Makefile#L210)
  LDFLAGS += -Wl,--no-undefined -Wl,--version-script=version.script

  [comms/ncclx/v2_27/src/version.script:11](../comms/ncclx/v2_27/src/version.script#L11)
  global: *nccl*; *ctran*; *cxa*;  // restrict export surface; local: *;

By contrast, upstream `thirdparty/nccl/src/Makefile` only builds its own subtrees and does not use a version script.

<a id="alltoallv-dataflow"></a>
## Dataflow: AllToAllv Path (NCCLX)

- Overview (when calling `ncclAllToAllv`):
  1) Validate pointers and counts.
  2) If `NCCL_ALLTOALLV_ALGO==ctran` and CTRAN is available, route to `ctranAllToAllv` (zero-copy path).
  3) Otherwise, fallback to baseline send/recv emulation.

  Flowchart (ASCII):

  +--------------+    algo==ctran &&    +-------------------+
  | ncclAllToAllv|----CTRX avail?------>| ctranAllToAllv(..) |
  +------+-------+      yes             +---+---------------+
         |                                ^
         | no                             |
         v                                |
  +------+------------------+             |
  | baseline send/recv path |-------------+
  +-------------------------+

  Key switch site:

  [comms/ncclx/v2_27/src/collectives.cc:397](../comms/ncclx/v2_27/src/collectives.cc#L397)
  if ((NCCL_ALLTOALLV_ALGO == NCCL_ALLTOALLV_ALGO::ctran) && ctranAllToAllvSupport(comm->ctranComm_.get())) {
    return metaCommToNccl(ctranAllToAllv(...));
  }

<a id="new-apis-flags"></a>
## How Are Additions Surfaced? New APIs and Flags

- New compile-time flags and symbols in the public header (feature detection):
  - `IS_NCCLX`, `NCCL_ALLTOALL(V)_SUPPORTED`, `NCCL_PERSISTENT_COLL_SUPPORTED`, `NCCL_RMA_SUPPORTED`, etc.
- New functions exported by libncclx:
  - Persistent collective APIs: `AllToAllInit`, `AllToAllExec`, `pFree`.
  - Hint plumbing: `setGlobalHint`, `getNcclxInfo`.
  - New collective: `ncclAllToAllv`.
  - Python module `ncclx_trainer_context` exposes training-related knobs.
  - CTRAN APIs via `CtranEx.h` and NCCL adapter via `CtranExComm`.
- New global symbol: `NCCL_COMM_WORLD`.
- Build/runtime integration hooks: extended logger/cvars, comms monitor, colltrace.

<a id="selected-diffs"></a>
## Representative Diffs (Selected Highlights)

1) Public header flags and new APIs:
- New in NCCLX:
  - [comms/ncclx/v2_27/src/nccl.h.in:19](../comms/ncclx/v2_27/src/nccl.h.in#L19)
  - [comms/ncclx/v2_27/src/nccl.h.in:37](../comms/ncclx/v2_27/src/nccl.h.in#L37)
  - [comms/ncclx/v2_27/src/nccl.h.in:52](../comms/ncclx/v2_27/src/nccl.h.in#L52)
  - [comms/ncclx/v2_27/src/nccl.h.in:58](../comms/ncclx/v2_27/src/nccl.h.in#L58)
  - [comms/ncclx/v2_27/src/nccl.h.in:13](../comms/ncclx/v2_27/src/nccl.h.in#L13) `AllToAllInit`
  - [comms/ncclx/v2_27/src/nccl.h.in:26](../comms/ncclx/v2_27/src/nccl.h.in#L26) `AllToAllExec`
  - [comms/ncclx/v2_27/src/nccl.h.in:34](../comms/ncclx/v2_27/src/nccl.h.in#L34) `pFree`
  - [comms/ncclx/v2_27/src/nccl.h.in:36](../comms/ncclx/v2_27/src/nccl.h.in#L36) `getNcclxInfo`
  - [comms/ncclx/v2_27/src/nccl.h.in:41](../comms/ncclx/v2_27/src/nccl.h.in#L41) `setGlobalHint`

2) Communicator world symbol (not in upstream):
- [comms/ncclx/v2_27/src/init.cc:66](../comms/ncclx/v2_27/src/init.cc#L66) defines `ncclCommWorld` and wiring at `:2048-2050`.

3) New collective implementation and routing:
- Switch point and implementation in [comms/ncclx/v2_27/src/collectives.cc:350](../comms/ncclx/v2_27/src/collectives.cc#L350)–[411](../comms/ncclx/v2_27/src/collectives.cc#L411).

4) Debug/logging integration with cvars:
- [comms/ncclx/v2_27/src/debug.cc:31](../comms/ncclx/v2_27/src/debug.cc#L31)–[55](../comms/ncclx/v2_27/src/debug.cc#L55) and `comms/utils/cvars/nccl_cvars.h:14+`.

5) Build/link visibility control:
- [comms/ncclx/v2_27/src/version.script:11](../comms/ncclx/v2_27/src/version.script#L11)–[16](../comms/ncclx/v2_27/src/version.script#L16) plus Makefile `LDFLAGS` at `:210-217`.

6) Env/param interface change:
- Upstream [thirdparty/nccl/src/include/param.h:12](../thirdparty/nccl/src/include/param.h#L12)–[29](../thirdparty/nccl/src/include/param.h#L29) vs NCCLX [comms/ncclx/v2_27/src/include/param.h:12](../comms/ncclx/v2_27/src/include/param.h#L12)–[15](../comms/ncclx/v2_27/src/include/param.h#L15).

<a id="feature-comparison"></a>
## Feature Comparison (Summary)

- NCCLX adds the following capability areas relative to upstream NCCL:
  - CTRAN transport and wrappers (AllGather/AllToAll(v) paths, direct and pipeline variants).
  - Persistent collectives (request-based init/exec, separate from stream semantics).
  - AllToAllv + dynamic split variants and non-contiguous support.
  - RMA window support flags and types.
  - Global communicator exposure (`NCCL_COMM_WORLD`).
  - Config via generated cvars and extended logging, plus comms monitor and colltrace.
  - Python bindings for trainer context / hints.

<a id="where-next"></a>
## Where to Explore Next

- Full diff: `codex/nccl_vs_ncclx_v2_27.diff` (14MB) and summary `codex/diff_quick_v2_27.txt`.
- New APIs code paths:
  - `comms/ncclx/v2_27/src/nccl.h.in` — public API surface and feature flags.
  - `comms/ncclx/v2_27/src/collectives.cc` — AllToAllv and persistent routing.
  - `comms/ncclx/v2_27/src/init.cc` — communicator initialization with CTRAN/colltrace/monitor.
  - `comms/ctran/CtranEx.h` and `comms/ncclx/v2_27/meta/wrapper/CtranExComm.h` — transport and adapter APIs.
  - `comms/ncclx/v2_27/src/version.script` — exported symbols policy.

<a id="appendix-invocation"></a>
## Appendix: Quick Invocation Map (ASCII)

- ncclAllToAllv (NCCLX):
  - [collectives.cc:397](../collectives.cc#L397) → if CTRAN enabled → `ctranAllToAllv(..)`; else baseline.
- Persistent AllToAllP:
  - [nccl.h.in:13](../nccl.h.in#L13) `AllToAllInit` → returns request → `AllToAllExec` on subsequent steps → `pFree`.
- Hints:
  - [nccl.h.in:41](../nccl.h.in#L41) `setGlobalHint(key,val)` — read by meta/ctran paths at init and runtime.

This report is a curated narrative. For a file-by-file comprehensive diff, open `codex/nccl_vs_ncclx_v2_27.diff` and `codex/diff_quick_v2_27.txt`.
