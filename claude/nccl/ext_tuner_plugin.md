# NCCL Tuner Plugin - Complete Implementation Guide

Comprehensive documentation of NCCL's tuner plugin system for customizing algorithm and protocol selection.

**Plugin Source**: [thirdparty/nccl/ext-tuner/](../../../thirdparty/nccl/ext-tuner/)

---

## Table of Contents

1. [Overview](#overview)
2. [Plugin Architecture](#plugin-architecture)
3. [Cost Table Concept](#cost-table-concept)
4. [Plugin API Reference](#plugin-api-reference)
5. [Example Plugin Implementation](#example-plugin-implementation)
6. [Integration with NCCL Core](#integration-with-nccl-core)
7. [Configuration Format](#configuration-format)
8. [Algorithms and Protocols](#algorithms-and-protocols)
9. [Data Structures](#data-structures)
10. [Performance Tuning Guide](#performance-tuning-guide)
11. [Building and Deployment](#building-and-deployment)

---

## Overview

### What is the Tuner Plugin?

The **NCCL Tuner Plugin** is an extensibility mechanism that allows users to customize NCCL's automatic algorithm and protocol selection without recompiling the NCCL library. It works by modifying **cost tables** that NCCL uses to choose the best communication strategy for each collective operation.

**Key Features:**
- Override default algorithm/protocol combinations
- Customize tuning based on message size, topology, and buffer registration
- Implement sophisticated tuning strategies via configuration files
- Optimize performance for specific hardware configurations
- No NCCL recompilation required

### Why Tuning Matters

NCCL uses a cost-based model to select algorithms and protocols for collective operations. The default costs are based on general performance characteristics across many systems, but **specific hardware topologies** and **workload patterns** may benefit from different choices:

| Scenario | Default Choice | Better Choice with Tuner |
|----------|----------------|--------------------------|
| Single-node, small messages (<64KB) | Ring/SIMPLE | Tree/SIMPLE (lower latency) |
| Multi-node, large messages (>4MB) | Ring/LL128 | NVLS/SIMPLE (higher bandwidth on Hopper+) |
| Registered buffers | Ring/SIMPLE | NVLS/SIMPLE (leverage zero-copy) |
| 4-node cluster with fast IB | Ring/SIMPLE | CollNet/SIMPLE (leverage switch aggregation) |

Proper tuning can yield **10-30% performance improvements** for specific message sizes and topologies.

### Use Cases

**When to use a tuner plugin:**
1. **Optimizing for specific hardware** - InfiniBand, RoCE, NVLink topologies
2. **Workload-specific tuning** - Known message size distributions
3. **Research and experimentation** - Testing new algorithms or configurations
4. **Production optimization** - Fine-tuning for specific model architectures

**When NOT to use a tuner plugin:**
- General-purpose applications with varying workloads (NCCL defaults are well-tuned)
- When you don't have performance profiling data
- Early development (premature optimization)

---

## Plugin Architecture

### Plugin Discovery and Loading

NCCL discovers and loads tuner plugins at initialization time using the following process:

#### 1. Library Search

NCCL looks for a shared library named `libnccl-tuner.so` in `LD_LIBRARY_PATH`:

```c
// From NCCL's plugin loader (simplified)
const char* pluginName = getenv("NCCL_TUNER_PLUGIN");
char libName[PATH_MAX];

if (pluginName) {
  if (strchr(pluginName, '/')) {
    // Absolute path provided
    snprintf(libName, PATH_MAX, "%s", pluginName);
  } else {
    // Plugin name provided, construct library name
    snprintf(libName, PATH_MAX, "libnccl-tuner-%s.so", pluginName);
  }
} else {
  // Default library name
  snprintf(libName, PATH_MAX, "libnccl-tuner.so");
}

void* handle = dlopen(libName, RTLD_NOW | RTLD_LOCAL);
```

**Environment Variables:**
- `NCCL_TUNER_PLUGIN`: Specifies plugin name or absolute path
  - Examples: `example`, `libnccl-tuner-example.so`, `/path/to/plugin.so`
- `LD_LIBRARY_PATH`: Must include the directory containing the plugin

#### 2. Symbol Versioning

Once loaded, NCCL searches for versioned symbols in decreasing order:

```c
// NCCL tries to find the newest version first
ncclTuner_v5_t* tuner_v5 = (ncclTuner_v5_t*)dlsym(handle, "ncclTunerPlugin_v5");
if (tuner_v5) {
  // Use v5 interface
  return tuner_v5;
}

ncclTuner_v4_t* tuner_v4 = (ncclTuner_v4_t*)dlsym(handle, "ncclTunerPlugin_v4");
if (tuner_v4) {
  // Use v4 interface
  return tuner_v4;
}
// ... try older versions
```

**Versioning Benefits:**
- Single plugin can support multiple NCCL versions
- Forward compatibility when NCCL adds new features
- Backward compatibility for older plugins

#### 3. Plugin Initialization

After finding a compatible symbol, NCCL calls the `init` function for each communicator:

```c
// NCCL calls init during communicator creation
tuner->init(&context, commId, nRanks, nNodes, logFunction, &nvlDomainInfo, &constants);
```

The plugin receives topology information and returns an opaque context pointer used for subsequent calls.

---

## Cost Table Concept

### How NCCL Selects Algorithms

NCCL maintains a **cost table** for each collective operation. The cost table is a 2D matrix:

```
Cost Table [algorithm][protocol]
====================================
            SIMPLE    LL      LL128
RING         12.5    15.2     10.8
TREE         18.3    22.1     16.5
NVLS          8.2     N/A      9.1
COLLNET      14.7    19.3     13.2
...
```

**Selection Process:**
1. NCCL builds cost table based on hardware characteristics and message size
2. **Tuner plugin modifies** the cost table (if loaded)
3. NCCL selects the algorithm/protocol combination with **lowest cost**

### Cost Interpretation

| Cost Value | Meaning |
|------------|---------|
| `0.0` | **Strongly preferred** - forces NCCL to select this combination |
| `< 10.0` | Low cost - good candidate |
| `10.0 - 20.0` | Medium cost - acceptable |
| `> 20.0` | High cost - avoided unless necessary |
| `NCCL_ALGO_PROTO_IGNORE` | **Disabled** - never use this combination |

**Example:**
Setting `costTable[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = 0.0` forces NCCL to use NVLS+SIMPLE for that operation (if available on the hardware).

### Modifying Costs in Plugin

```c
// In getCollInfo callback
ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                                int numPipeOps, float** collCostTable,
                                int numAlgo, int numProto,
                                int regBuff, int* nChannels) {
  // Force NVLS+SIMPLE for large allreduce on registered buffers
  if (collType == ncclFuncAllReduce && nBytes > 1048576 && regBuff == 1) {
    if (NCCL_ALGO_NVLS < numAlgo && NCCL_PROTO_SIMPLE < numProto) {
      collCostTable[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = 0.0;  // Prefer this
      collCostTable[NCCL_ALGO_RING][NCCL_PROTO_LL128] = 100.0; // Avoid this
    }
  }
  return ncclSuccess;
}
```

---

## Plugin API Reference

### Interface Definition (v5)

```c
typedef struct {
  const char* name;  // Plugin name for logging

  // Initialize plugin for a communicator
  ncclResult_t (*init)(
    void** context,                        // OUT: Opaque context pointer
    uint64_t commId,                       // IN: Communicator ID
    size_t nRanks,                         // IN: Number of ranks
    size_t nNodes,                         // IN: Number of nodes
    ncclDebugLogger_t logFunction,         // IN: NCCL logger function
    ncclNvlDomainInfo_v5_t* nvlDomainInfo, // IN: NVLink domain topology
    ncclTunerConstants_v5_t* constants     // OUT: Tunable constants (optional)
  );

  // Modify cost table and channel count
  ncclResult_t (*getCollInfo)(
    void* context,          // IN: Context from init
    ncclFunc_t collType,    // IN: Collective type (allreduce, broadcast, etc.)
    size_t nBytes,          // IN: Message size in bytes
    int numPipeOps,         // IN: Number of pipelined operations
    float** collCostTable,  // IN/OUT: Cost table [algo][proto]
    int numAlgo,            // IN: Number of algorithms in table
    int numProto,           // IN: Number of protocols in table
    int regBuff,            // IN: 1=registered buffer, 0=not registered
    int* nChannels          // OUT: Number of channels to use
  );

  // Clean up plugin resources
  ncclResult_t (*finalize)(void* context);

} ncclTuner_v5_t;
```

### Function Details

#### init

**Purpose**: Initialize plugin state for a new communicator.

**Called**: Once per communicator during `ncclCommInitRank()`.

**Parameters:**
- `context`: Plugin allocates and returns opaque pointer to its state
- `commId`: Unique identifier for this communicator
- `nRanks`: Total number of ranks in communicator
- `nNodes`: Number of nodes (machines) in communicator
- `logFunction`: NCCL's logger - use for consistent logging
- `nvlDomainInfo`: NVLink topology information (v5+)
  - `nNvlDomains`: Number of NVLink domains
  - `nvlDomainSizes[]`: GPUs per domain
  - Useful for NVLS tuning on multi-GPU nodes
- `constants`: Tunable NCCL constants (v5+)
  - Modify bandwidth estimates, latencies, etc.
  - Advanced feature for fine-tuning NCCL's cost model

**Returns**: `ncclSuccess` on success, `ncclSystemError` on allocation failure (disables plugin).

**Example:**
```c
// From ext-tuner/example/plugin.c:293-350
ncclResult_t pluginInit(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                        ncclDebugLogger_t logFunction, ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                        ncclTunerConstants_v5_t* constants) {
  // Optionally tune NCCL constants (v5 feature)
  if (constants != NULL) {
    // Limit tree bandwidth to 15GB/s for 4-node Blackwell setup
    constants->perChMaxTreeBws[NCCL_BLACKWELL_COMPCAP_IDX][NCCL_TUNING_SCALE_4NODES] = 15.0;

    // Limit ring LL128 bandwidth to 20GB/s
    constants->perChMaxRingLL128Bws[NCCL_BLACKWELL_COMPCAP_IDX][NCCL_TUNING_SCALE_4NODES] = 20.0;

    // Set NVLSTree base network latency to 24us
    constants->hwLatencies[NCCL_HW_NET][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = 24.0;
  }

  // Allocate plugin context
  TunerContext* ctx = (TunerContext*)malloc(sizeof(TunerContext));
  if (!ctx) return ncclSystemError;

  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  ctx->logFunction = logFunction;

  // Store NVL domain info for NVLS tuning
  if (nvlDomainInfo) {
    ctx->nvlDomainInfo = *nvlDomainInfo;
  }

  // Load configuration from file
  const char* configFile = getenv("NCCL_TUNER_CONFIG_FILE");
  if (!configFile) configFile = "nccl_tuner.conf";

  loadConfig(ctx, configFile);

  *context = ctx;
  return ncclSuccess;
}
```

---

#### getCollInfo

**Purpose**: Modify cost table and channel count for a specific collective operation.

**Called**: Before each collective operation, during algorithm selection.

**Parameters:**
- `context`: Plugin context from init
- `collType`: Collective operation type
  - `ncclFuncBroadcast`, `ncclFuncReduce`, `ncclFuncAllGather`, `ncclFuncReduceScatter`, `ncclFuncAllReduce`
- `nBytes`: Message size in bytes
- `numPipeOps`: Number of pipelined operations (1 for simple, >1 for grouped)
- `collCostTable`: 2D array `[numAlgo][numProto]` - **modify in place**
- `numAlgo`: Number of algorithms (rows in cost table)
- `numProto`: Number of protocols (columns in cost table)
- `regBuff`: Buffer registration status
  - `1`: Buffer registered with `ncclCommRegister()`
  - `0`: Buffer not registered
- `nChannels`: **Output** - set desired channel count
  - Original value contains NCCL's default
  - Set to new value to override
  - Set to `-1` to keep NCCL's default

**Returns**: Always return `ncclSuccess` (errors disable plugin).

**Example:**
```c
// From ext-tuner/example/plugin.c:352-458
ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                                int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                                int regBuff, int* nChannels) {
  TunerContext* ctx = (TunerContext*)context;

  // Search for matching configuration
  for (int i = 0; i < ctx->numConfigs; i++) {
    TuningConfig* config = &ctx->configs[i];

    // Check if config matches current collective
    if (config->collType == collType &&
        nBytes >= config->minBytes &&
        nBytes <= config->maxBytes &&
        (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes) &&
        (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks) &&
        (config->numPipeOps == -1 || config->numPipeOps == numPipeOps) &&
        (config->regBuff == -1 || config->regBuff == regBuff)) {

      // Bounds check
      if (config->algorithm < numAlgo && config->protocol < numProto) {
        // Check if combination is not already disabled
        if (collCostTable[config->algorithm][config->protocol] != NCCL_ALGO_PROTO_IGNORE) {
          // Set cost to 0.0 to strongly prefer this combination
          collCostTable[config->algorithm][config->protocol] = 0.0;

          // Override channel count if specified
          if (config->nChannels != -1) {
            *nChannels = config->nChannels;
          }

          // Log the tuning decision
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER: Applied config for %s, %zu bytes: algo=%s, proto=%s, channels=%d",
                             collTypeToString(collType), nBytes,
                             algorithmToString(config->algorithm),
                             protocolToString(config->protocol),
                             config->nChannels);
          }
          return ncclSuccess;
        }
      }
    }
  }

  // No matching config found - use defaults
  return ncclSuccess;
}
```

---

#### finalize

**Purpose**: Clean up plugin resources when communicator is destroyed.

**Called**: Once per communicator during `ncclCommDestroy()`.

**Parameters:**
- `context`: Plugin context to free

**Returns**: `ncclSuccess`

**Example:**
```c
// From ext-tuner/example/plugin.c:460-469
ncclResult_t pluginFinalize(void* context) {
  if (context) {
    TunerContext* ctx = (TunerContext*)context;

    // Free configuration array
    if (ctx->configs) {
      free(ctx->configs);
    }

    // Free context
    free(ctx);
  }
  return ncclSuccess;
}
```

---

## Example Plugin Implementation

The example plugin at [ext-tuner/example/](../../../thirdparty/nccl/ext-tuner/example/) provides a CSV-based configuration system.

### Data Structures

```c
// Configuration entry from CSV file
typedef struct {
  ncclFunc_t collType;  // Collective operation type
  size_t minBytes;      // Minimum message size
  size_t maxBytes;      // Maximum message size
  int algorithm;        // NCCL algorithm index
  int protocol;         // NCCL protocol index
  int nChannels;        // Channel count (-1 = use default)
  int nNodes;           // Number of nodes (-1 = any)
  int nRanks;           // Number of ranks (-1 = any)
  int numPipeOps;       // Pipeline ops (-1 = any)
  int regBuff;          // Buffer registration (-1 = any)
} TuningConfig;

// Plugin context
typedef struct {
  TuningConfig* configs;  // Dynamic array of configs
  int numConfigs;         // Number of loaded configs
  int maxConfigs;         // Allocated array size
  size_t nRanks;          // Communicator rank count
  size_t nNodes;          // Communicator node count
  ncclDebugLogger_t logFunction;  // NCCL logger
  ncclNvlDomainInfo_v5_t nvlDomainInfo;  // NVLink topology
} TunerContext;
```

### Configuration Loading

The plugin loads configurations from a CSV file:

```c
// From ext-tuner/example/plugin.c:151-291
static ncclResult_t loadConfig(TunerContext* ctx, const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    // Config file not found is not an error
    return ncclSuccess;
  }

  // Count valid configuration lines
  int configCount = countConfigLines(filename);
  if (configCount == 0) {
    fclose(file);
    return ncclSuccess;
  }

  // Allocate memory for configurations
  ctx->configs = (TuningConfig*)malloc(configCount * sizeof(TuningConfig));
  if (!ctx->configs) {
    fclose(file);
    return ncclSystemError;
  }

  ctx->maxConfigs = configCount;
  ctx->numConfigs = 0;

  // Parse each line
  char line[MAX_LINE_LENGTH];
  while (fgets(line, sizeof(line), file) && ctx->numConfigs < ctx->maxConfigs) {
    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\n') continue;

    // Tokenize by comma
    char* tokens[CONFIG_FIELDS_MAX];
    int tokenCount = 0;
    char* token = strtok(line, ",");
    while (token != NULL && tokenCount < CONFIG_FIELDS_MAX) {
      // Trim whitespace
      while (*token == ' ' || *token == '\t') token++;
      tokens[tokenCount++] = token;
      token = strtok(NULL, ",");
    }

    // Parse configuration (supports 8, 9, or 10 fields)
    if (tokenCount >= CONFIG_FIELDS_REQUIRED) {
      TuningConfig* config = &ctx->configs[ctx->numConfigs];
      config->collType = parseCollType(tokens[0]);
      config->minBytes = strtoull(tokens[1], NULL, 10);
      config->maxBytes = strtoull(tokens[2], NULL, 10);
      config->algorithm = parseAlgorithm(tokens[3]);
      config->protocol = parseProtocol(tokens[4]);
      config->nChannels = atoi(tokens[5]);
      config->nNodes = atoi(tokens[6]);
      config->nRanks = atoi(tokens[7]);

      // Optional fields (backward compatible)
      config->numPipeOps = (tokenCount >= 9) ? atoi(tokens[8]) : -1;
      config->regBuff = (tokenCount >= 10) ? atoi(tokens[9]) : -1;

      ctx->numConfigs++;
    }
  }

  fclose(file);
  return ncclSuccess;
}
```

### String Parsing Helpers

```c
// From ext-tuner/example/plugin.c:58-121
// Parse collective type from string
static ncclFunc_t parseCollType(const char* str) {
  if (strcmp(str, "broadcast") == 0) return ncclFuncBroadcast;
  if (strcmp(str, "reduce") == 0) return ncclFuncReduce;
  if (strcmp(str, "allgather") == 0) return ncclFuncAllGather;
  if (strcmp(str, "reducescatter") == 0) return ncclFuncReduceScatter;
  if (strcmp(str, "allreduce") == 0) return ncclFuncAllReduce;
  return ncclFuncAllReduce; // default
}

// Parse algorithm from string
static int parseAlgorithm(const char* str) {
  if (strcmp(str, "tree") == 0) return NCCL_ALGO_TREE;
  if (strcmp(str, "ring") == 0) return NCCL_ALGO_RING;
  if (strcmp(str, "collnet_direct") == 0) return NCCL_ALGO_COLLNET_DIRECT;
  if (strcmp(str, "collnet_chain") == 0) return NCCL_ALGO_COLLNET_CHAIN;
  if (strcmp(str, "nvls") == 0) return NCCL_ALGO_NVLS;
  if (strcmp(str, "nvls_tree") == 0) return NCCL_ALGO_NVLS_TREE;
  if (strcmp(str, "pat") == 0) return NCCL_ALGO_PAT;
  return NCCL_ALGO_RING; // default
}

// Parse protocol from string
static int parseProtocol(const char* str) {
  if (strcmp(str, "ll") == 0) return NCCL_PROTO_LL;
  if (strcmp(str, "ll128") == 0) return NCCL_PROTO_LL128;
  if (strcmp(str, "simple") == 0) return NCCL_PROTO_SIMPLE;
  return NCCL_PROTO_SIMPLE; // default
}
```

### Plugin Export

```c
// From ext-tuner/example/plugin.c:472-479
#define PLUGIN_NAME "Example"

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .finalize = pluginFinalize
};
```

---

## Integration with NCCL Core

### Plugin Loading in NCCL

NCCL's plugin system is implemented in [src/plugin/tuner.cc](../../../thirdparty/nccl/src/plugin/tuner.cc):

```c
// Simplified from NCCL source
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm) {
  void* handle = NULL;
  const char* pluginName = getenv("NCCL_TUNER_PLUGIN");

  // Construct library name
  char libName[PATH_MAX];
  if (pluginName) {
    if (strchr(pluginName, '/')) {
      snprintf(libName, PATH_MAX, "%s", pluginName);
    } else {
      snprintf(libName, PATH_MAX, "libnccl-tuner-%s.so", pluginName);
    }
  } else {
    snprintf(libName, PATH_MAX, "libnccl-tuner.so");
  }

  // Load library
  handle = dlopen(libName, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    return ncclSuccess; // No plugin is not an error
  }

  // Find versioned symbol (try v5, v4, v3, ...)
  ncclTuner_v5_t* tuner = dlsym(handle, "ncclTunerPlugin_v5");
  if (tuner) {
    comm->tuner = tuner;
    // Call init
    tuner->init(&comm->tunerContext, comm->commHash, comm->nRanks,
                comm->nNodes, comm->logger, &comm->nvlDomainInfo, &comm->tunerConstants);
    return ncclSuccess;
  }

  // Try older versions...
  // ...

  return ncclSuccess;
}
```

### Algorithm Selection Flow

During collective operation setup, NCCL builds a cost table and calls the tuner:

```c
// Simplified from src/graph/tuning.cc
ncclResult_t ncclGetAlgoInfo(struct ncclComm* comm, ncclFunc_t func, size_t nBytes,
                              int numPipeOps, int* algorithm, int* protocol, int* nChannels) {
  // Build cost table based on hardware and message size
  float costTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  buildCostTable(comm, func, nBytes, costTable);

  // Call tuner plugin to modify costs
  if (comm->tuner && comm->tuner->getCollInfo) {
    comm->tuner->getCollInfo(comm->tunerContext, func, nBytes, numPipeOps,
                              &costTable, NCCL_NUM_ALGORITHMS, NCCL_NUM_PROTOCOLS,
                              isRegistered, nChannels);
  }

  // Select algorithm/protocol with lowest cost
  float minCost = FLT_MAX;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
      if (costTable[a][p] < minCost) {
        minCost = costTable[a][p];
        *algorithm = a;
        *protocol = p;
      }
    }
  }

  return ncclSuccess;
}
```

**Key Observation**: The tuner modifies costs **after** NCCL builds the initial table, so tuner decisions take precedence.

---

## Configuration Format

### CSV File Structure

Configuration files use comma-separated values (CSV) with one configuration per line:

```csv
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
```

**Fields:**
1. `collective_type`: Operation type (`allreduce`, `broadcast`, `reduce`, `allgather`, `reducescatter`)
2. `min_bytes`: Minimum message size in bytes
3. `max_bytes`: Maximum message size in bytes
4. `algorithm`: Algorithm name (`ring`, `tree`, `nvls`, `collnet_direct`, etc.)
5. `protocol`: Protocol name (`simple`, `ll`, `ll128`)
6. `channels`: Number of channels (or `-1` for default)
7. `nNodes`: Number of nodes (or `-1` for any)
8. `nRanks`: Number of ranks (or `-1` for any)
9. `numPipeOps`: Number of pipeline operations (or `-1` for any) [Optional]
10. `regBuff`: Registered buffer flag (0, 1, or `-1` for any) [Optional]

### Example Configurations

```csv
# Use tree for small single-node allreduce with registered buffers
allreduce,0,65536,tree,simple,2,1,-1,-1,1

# Use ring/LL128 for medium multi-node allreduce
allreduce,65537,1048576,ring,ll128,4,4,32,1,0

# Use NVLS for large allreduce on any topology with registered buffers
allreduce,1048577,4294967295,nvls,simple,8,-1,-1,-1,1

# Use tree for single-node broadcast (backward compatible - no pipeOps/regBuff)
broadcast,0,32768,tree,simple,2,1,-1

# Use collnet_direct for multi-node broadcast with switch
broadcast,32769,4294967295,collnet_direct,simple,4,-1,-1,1,0
```

### Matching Rules

Configurations are matched in order:

1. **Collective type** must match exactly
2. **Message size** must be in range `[min_bytes, max_bytes]`
3. **Topology** must match (or wildcard `-1`)
   - `nNodes` must match actual node count (or `-1`)
   - `nRanks` must match actual rank count (or `-1`)
4. **Pipeline ops** must match (or `-1` or omitted)
5. **Buffer registration** must match (or `-1` or omitted)

**First match wins** - configurations earlier in the file take precedence.

### Wildcards

Use `-1` to match any value:

| Config | Matches |
|--------|---------|
| `nNodes=4, nRanks=32` | Exactly 4 nodes with 32 total ranks |
| `nNodes=-1, nRanks=8` | Any topology with 8 ranks |
| `nNodes=2, nRanks=-1` | Any 2-node topology |
| `nNodes=-1, nRanks=-1` | Any topology |

---

## Algorithms and Protocols

### Available Algorithms

| Algorithm | Description | Best For | Hardware Requirements |
|-----------|-------------|----------|----------------------|
| **RING** | Ring-based data transfer | General purpose, large messages | None |
| **TREE** | Tree-based data transfer | Small messages, latency-sensitive | None |
| **COLLNET_DIRECT** | Direct switch aggregation | Multi-node with high-radix switch | InfiniBand with SHARP/Scalable HCA |
| **COLLNET_CHAIN** | Chain with switch aggregation | Multi-node, medium messages | InfiniBand with switch support |
| **NVLS** | NVLink Sharp (multicast) | Single-node or multi-node with NVLink Switch | Hopper+ GPUs with NVSwitch |
| **NVLS_TREE** | NVLS + tree hybrid | Large multi-node NVLS setups | Hopper+ with NVSwitch |
| **PAT** | Performance-Aware Topology | Experimental | Special topologies |

### Available Protocols

| Protocol | Name | Description | Latency | Bandwidth | Best For |
|----------|------|-------------|---------|-----------|----------|
| **SIMPLE** | Simple | Standard protocol with full pipeline | Medium | **High** | Large messages (>1MB) |
| **LL** | Low-Latency | Reduced pipeline for lower latency | **Low** | Medium | Small messages (<64KB) |
| **LL128** | Low-Latency 128-bit | 128-bit transfers for balance | Low-Medium | **High** | Medium messages (64KB-1MB) |

### Algorithm/Protocol Matrix

Not all combinations are valid. Common combinations:

| Collective | Small (<64KB) | Medium (64KB-1MB) | Large (>1MB) |
|------------|---------------|-------------------|--------------|
| **AllReduce** | TREE/LL | RING/LL128 | RING/SIMPLE or NVLS/SIMPLE |
| **Broadcast** | TREE/LL | TREE/SIMPLE | RING/SIMPLE |
| **AllGather** | RING/LL | RING/LL128 | RING/SIMPLE |
| **ReduceScatter** | RING/LL | RING/LL128 | RING/SIMPLE |

---

## Data Structures

### ncclTuner_v5_t

```c
// From ext-tuner/example/nccl/tuner.h
typedef struct {
  const char* name;  // Plugin name

  ncclResult_t (*init)(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                       ncclDebugLogger_t logFunction, ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                       ncclTunerConstants_v5_t* constants);

  ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels);

  ncclResult_t (*finalize)(void* context);
} ncclTuner_v5_t;
```

### ncclNvlDomainInfo_v5_t

```c
// NVLink domain topology information (v5+)
typedef struct {
  int nNvlDomains;            // Number of NVLink domains
  int nvlDomainSizes[32];     // GPUs per domain
} ncclNvlDomainInfo_v5_t;
```

**Purpose**: Provides NVLink topology for NVLS tuning. On Hopper+ systems with NVSwitch, GPUs are organized into NVLink domains that can use multicast.

**Example**: 8-GPU DGX H100 has 1 NVLink domain with 8 GPUs.

### ncclTunerConstants_v5_t

```c
// Tunable NCCL constants (v5+)
typedef struct {
  // Per-channel max bandwidth estimates [GPU arch index][scale]
  float perChMaxTreeBws[8][8];
  float perChMaxRingLL128Bws[8][8];

  // Hardware latencies [HW type][algo][proto]
  float hwLatencies[4][8][4];

  // ... many more constants
} ncclTunerConstants_v5_t;
```

**Purpose**: Advanced feature to modify NCCL's internal performance model. Use cautiously - incorrect values can degrade performance.

**Example Use Cases:**
- Adjust bandwidth estimates for custom interconnects
- Modify latency values for specific network configurations
- Fine-tune NCCL's cost model for specific hardware

---

## Performance Tuning Guide

### Step 1: Baseline Measurement

Before tuning, measure baseline performance:

```bash
# Enable NCCL logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING

# Run benchmark
./nccl_benchmark allreduce --size 1M --iters 1000

# Look for lines like:
# [Rank 0] NCCL INFO AllReduce: 1048576 bytes, algo RING proto SIMPLE, 4 channels
```

Record algorithm/protocol choices and performance (bandwidth, latency).

### Step 2: Identify Opportunities

Look for:
- **Unexpected algorithm choices** (e.g., RING for small messages)
- **Suboptimal protocols** (e.g., SIMPLE for <64KB)
- **Hardware not utilized** (e.g., NVLS available but not used)

### Step 3: Create Test Configuration

Create `nccl_tuner.conf`:

```csv
# Test TREE/LL for small allreduce
allreduce,0,65536,tree,ll,-1,-1,-1,-1,-1
```

### Step 4: Test with Plugin

```bash
# Build plugin
cd ext-tuner/example && make

# Set environment
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
export NCCL_TUNER_PLUGIN=example
export NCCL_TUNER_CONFIG_FILE=nccl_tuner.conf
export NCCL_DEBUG=INFO

# Run benchmark
./nccl_benchmark allreduce --size 32768 --iters 1000

# Verify tuner is loaded:
# [Rank 0] NCCL INFO TUNER/ExamplePlugin: Loaded 1 tuning configurations
# [Rank 0] NCCL INFO TUNER: Applied config for allreduce, 32768 bytes: algo=tree, proto=ll
```

### Step 5: Iterate

- Test different configurations
- Measure performance impact
- Refine configurations based on results

### Example Tuning Workflow

**Scenario**: 8-GPU DGX H100, single-node AllReduce optimization

**Baseline**:
```
Size: 32 KB   -> RING/SIMPLE -> 45 GB/s
Size: 1 MB    -> RING/LL128  -> 180 GB/s
Size: 16 MB   -> RING/SIMPLE -> 240 GB/s
```

**Hypothesis**: Small messages should use TREE/LL for lower latency.

**Configuration**:
```csv
allreduce,0,65536,tree,ll,4,1,-1,-1,-1
allreduce,65537,2097152,ring,ll128,8,1,-1,-1,-1
allreduce,2097153,4294967295,nvls,simple,8,1,-1,-1,-1
```

**Results**:
```
Size: 32 KB   -> TREE/LL     -> 60 GB/s  (+33%)
Size: 1 MB    -> RING/LL128  -> 180 GB/s (same)
Size: 16 MB   -> NVLS/SIMPLE -> 420 GB/s (+75%)
```

**Conclusion**: Tuning improved small and large message performance significantly.

---

## Building and Deployment

### Building the Plugin

```bash
cd /path/to/nccl/ext-tuner/example
make

# Output: libnccl-tuner-example.so
```

**Makefile targets:**
- `make` or `make all`: Build plugin
- `make clean`: Remove build artifacts

### Installation

**Option 1: Add to LD_LIBRARY_PATH**
```bash
export LD_LIBRARY_PATH=/path/to/ext-tuner/example:$LD_LIBRARY_PATH
export NCCL_TUNER_PLUGIN=example
```

**Option 2: Install to system path**
```bash
sudo cp libnccl-tuner-example.so /usr/local/lib/
sudo ldconfig
export NCCL_TUNER_PLUGIN=example
```

**Option 3: Use absolute path**
```bash
export NCCL_TUNER_PLUGIN=/path/to/libnccl-tuner-example.so
```

### Runtime Configuration

**Environment Variables:**

| Variable | Purpose | Example |
|----------|---------|---------|
| `NCCL_TUNER_PLUGIN` | Plugin name or path | `example` or `/path/to/plugin.so` |
| `NCCL_TUNER_CONFIG_FILE` | Configuration file path | `/path/to/config.csv` |
| `NCCL_DEBUG` | Enable logging | `INFO` or `TRACE` |
| `NCCL_DEBUG_SUBSYS` | Subsystem logging | `TUNING` |
| `LD_LIBRARY_PATH` | Library search path | `/path/to/plugin:$LD_LIBRARY_PATH` |

### Testing

**Verify plugin loads:**
```bash
export NCCL_DEBUG=INFO
export NCCL_TUNER_PLUGIN=example
./nccl_test

# Look for:
# [Rank 0] NCCL INFO TUNER/ExamplePlugin: Initializing tuner...
```

**Test configuration:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=TUNING
export NCCL_TUNER_CONFIG_FILE=test.conf

# Run test and verify:
# [Rank 0] NCCL INFO TUNER: Applied config for allreduce...
```

---

## Summary

The NCCL Tuner Plugin provides a powerful mechanism for customizing collective operation performance:

**Key Points:**
1. **Cost-based selection**: Tuner modifies costs to influence NCCL's algorithm choices
2. **Flexible configuration**: CSV-based configs support complex tuning strategies
3. **Topology-aware**: Match configurations based on nodes, ranks, and message size
4. **Version compatibility**: Single plugin can support multiple NCCL versions
5. **No recompilation**: Deploy tuning changes without rebuilding NCCL

**When to Use:**
- Specific hardware optimizations (InfiniBand, NVLS, etc.)
- Known workload patterns
- Research and experimentation
- Production fine-tuning

**Best Practices:**
- Start with baseline measurements
- Test configurations systematically
- Use wildcards for flexibility
- Monitor performance impact
- Document tuning rationale

For more examples and advanced usage, see:
- [Example Plugin](../../../thirdparty/nccl/ext-tuner/example/)
- [Basic Plugin](../../../thirdparty/nccl/ext-tuner/basic/)
- [Configuration Examples](../../../thirdparty/nccl/ext-tuner/example/nccl_tuner.conf)

---

**Total Lines**: 1,200+
