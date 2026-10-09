# Shared/private allocation throughput on Jetson AGX Xavier

```sh
make memory_cost
./memory_cost                       # 64 MiB, 10 s per case, 4 CPU workers
make run-memory-sweep               # 64 KiB and 64 MiB; about 80 s total
REPEATS=3 sh ./run-memory-cost.sh       # about 240 s; alternate case order
PATTERN=random sh ./run-memory-cost.sh # optional random locations
```

The Makefile uses `/usr/local/cuda-11.4/bin/nvcc`, C++11 and `sm_72`.
Override it with `make NVCC=/another/path/nvcc memory_cost`.
No sudo is needed if your account can already run the preceding CUDA tests.

The four cases compare:

| Side performing updates | Shared allocation | Private allocation |
|---|---|---|
| CPU only | `cudaHostAlloc(..., cudaHostAllocMapped)` + `cudaHostGetDevicePointer(..., 0)` | `posix_memalign(..., 4096, ...)`, ordinary pageable host memory |
| GPU only | Exactly the same mapped-host allocation and flags | `cudaMalloc` |

The allocation calls and flags for shared data match the preceding atomicity
and ordering tests. `cudaDeviceScheduleBlockingSync` is added to the device
initialization flags to let the host sleep during GPU timing waits; it does
not change the shared allocation flags. No write-combining or managed-memory
flags are added.

Each case has a warmup and then at least 10 seconds of measured updates.
Allocation, initialization, warmup, verification and freeing are excluded.
During CPU measurements no GPU kernel runs. During GPU measurements the CPU
only submits work and waits for CUDA events; it never accesses the test buffer.
There is no ownership transfer or ping-pong in the measured interval.

CPU workers are pinned to the first four CPUs allowed by the process's affinity
mask and own separate, cache-line-aligned partitions. GPU work uses four blocks
per SM, 128 threads per block; on Xavier's eight-SM GPU this is 4096 threads.
Each GPU thread owns a disjoint stripe. Thus ordinary updates have no data
races. Sequential GPU accesses are coalesced; optional random accesses use a
per-thread xorshift generator within each thread's own stripe. Random CPU
accesses stay within each worker's contiguous partition. Random results
include index-generation and changed coalescing costs; compare allocation
types within the same side and pattern.

The operation is a 32-bit unsigned load/add/store (`a[index]++`), with no atomic
instructions or hardware fences in the update loop. CPU accesses are volatile
so repeated modifications cannot become one accumulated store. GPU inline PTX
uses ordinary `ld.global.cg.u32` and `st.global.wb.u32` with compiler barriers
so every increment retains a load and store. The same instructions run for both
GPU allocations. `.cg` asks for L2 caching and bypasses GPU L1 where the memory
mapping permits caching; it does not override an uncached allocation mapping.
There is no forced DRAM flush after each update: cached private data may stay
in cache, which is exactly the allocation-policy effect being measured.

GPU batches adapt to about 20 ms each. Timing uses actual elapsed time, so a
case can overrun its requested duration by a final batch. CUDA events also
report accumulated GPU execution time. Host event waits block instead of
polling the shared data. The CSV's primary `Mupdates_per_s` uses wall time for
both sides; `GPU_active_Mupdates_per_s` excludes host launch gaps. A post-run
checksum must equal the counted updates or the program exits with an error.
Very long runs can overflow an individual 32-bit counter; checksum checking
will detect that instead of reporting success.

Arguments:

```text
./memory_cost [MiB=64] [seconds=10] [CPU_threads=4] [sequential|random] [repeats=1]
```

Sizes must be powers of two between 0.0625 and 4096 MiB; CPU worker counts are
1, 2, 4 or 8, subject to the process's affinity mask. Each repeat executes
all four cases. Odd repeats run private then shared; even repeats reverse
the order. The script's environment variables are `SECONDS_PER_CASE`,
`CPU_THREADS`, `PATTERN` and `REPEATS`. CSV rows go to stdout and configuration,
progress and shared/private ratios to stderr. For example:

```sh
./memory_cost 0.0625 10 4 sequential 3 > small.csv 2> small.log
./memory_cost 64 10 4 sequential 3 > large.csv 2> large.log
```

`shared/private=0.5` means shared throughput is half the private baseline.
`shared throughput change=-50%` expresses the same result. Throughput near
1.0 means no measurable penalty under this workload and duration; use the
repetitions to check noise before treating a small difference as meaningful.

## What this establishes

This measures the **total steady-state allocation-policy cost with the other
processor inactive**. It does not isolate coherence-protocol traffic from
cacheability, translation, page size or memory-mapping effects, and it does
not measure coherence under concurrent CPU/GPU access. Allocation, launch
handoff and final visibility-maintenance costs are outside its timed updates.

NVIDIA's CUDA 11.4 Tegra documentation lists mapped pinned host memory as
CPU-cached on Xavier (SM 7.2) but GPU-uncached, whereas device memory is
GPU-cached. Consequently a small working set is useful for exposing GPU
cacheability differences, and a large one measures behavior beyond cache
capacity. CPU shared/private results may be similar because both allocations
are CPU-cached. A GPU penalty is consistent with the documented uncached
mapping; this benchmark alone cannot attribute its entire magnitude to
the coherence protocol.

Sources:

- https://docs.nvidia.com/cuda/archive/11.4.4/cuda-for-tegra-appnote/index.html#memory-management
- https://docs.nvidia.com/cuda/archive/11.4.4/parallel-thread-execution/index.html#cache-operators

## Validation build without CUDA

`make memory_cost_cpu_private` builds only the real CPU-private path with g++.
This is useful for checking timing, thread partitioning, affinity and checksums;
it prints its restricted mode explicitly and produces no shared/GPU results.
It does not validate GPU compilation or execution.
