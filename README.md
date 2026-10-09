# Jetson AGX Xavier CPU–GPU ordering tests

On the deployed Xavier node, with your account in the `video` group:

```sh
make
make run-ordering
```

The Makefile uses `/usr/local/cuda-11.4/bin/nvcc`; override with
`make NVCC=/another/path/nvcc` if needed. The earlier atomicity tests remain
available as `make run` and `make run-no-nvcc`.

## Shared/private throughput

`make memory_cost` builds the additional allocation-policy benchmark.
`make run-memory-sweep` compares CPU-only and GPU-only updates to shared and
private buffers with 64 KiB and 64 MiB working sets, 10 seconds per case.
See [MEMORY_COST.md](MEMORY_COST.md) for commands, timing and interpretation.

## CPU/GPU ordering

```sh
make run-ordering-cpu-to-gpu
make run-ordering-gpu-to-cpu
# Or run both, consecutively (about 20 seconds total):
make run-ordering
```

Each direction has four independent `(a, b)` pairs, with one writer per pair.
The writer performs ordinary `a++`, a system fence, then ordinary `b++`. The
reader loads `b`, performs a system fence, loads `a`, and checks `a >= b`.
The CPU fence is `__sync_synchronize()`; the GPU fence is
`__threadfence_system()`. The shared fields are `volatile` to force actual
loads/stores. **No atomic operations are used by these ordering tests.**

Each test uses pinned mapped host memory and runs for 10 seconds. It requires
both sides to make progress in at least 8 of 10 one-second intervals and
reports counts, `a < b` observations, and final values. Exit status is 0 for
an observed pass, 1 for a violation, 2 for error or insufficient overlap,
and 124 for the external timeout.

This is a hardware litmus, not a conforming C++ synchronization pattern:
concurrent ordinary reads and writes are data races in C++. `volatile` and
fences do not fix that language-level issue. A pass describes what happened
on this Xavier and allocation over these 10 seconds; it does not establish
a portable guarantee. For application synchronization, use system-scope
release/acquire atomics.

## Earlier atomicity test

`make run` runs the earlier 10-second test of a shared 64-bit atomic counter:
four CPU threads and 128 GPU threads increment it concurrently, and the
result is compared with independently counted contributions. The separate
`make run-no-nvcc` target runs the older short driver-based version.
