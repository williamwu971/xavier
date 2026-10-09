// Xavier allocation-policy throughput: one processor accesses the buffer at a time.
// CPU-only validation: g++ -x c++ -DCPU_PRIVATE_ONLY -O2 -std=c++11 -pthread ...
#ifndef CPU_PRIVATE_ONLY
#include <cuda_runtime.h>
#endif
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>

using Clock = std::chrono::steady_clock;
using Word = uint32_t;
struct Result { uint64_t ops; double seconds, gpu_seconds; };
static void fail(const char *message) {
    std::fprintf(stderr, "ERROR: %s\n", message);
    std::exit(2);
}
#ifndef CPU_PRIVATE_ONLY
#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); \
    std::exit(2); } } while (0)
#endif
static bool power2(size_t n) { return n && !(n & (n - 1)); }
static uint32_t random_next(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
static uint64_t checksum(const Word *a, size_t n) {
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) sum += a[i];
    return sum;
}
static std::vector<int> cpu_ids() {
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask)) fail("sched_getaffinity failed");
    std::vector<int> ids;
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &mask)) ids.push_back(i);
    return ids;
}

// Disjoint, cache-line-aligned partitions; no atomics/fences in the update loop.
static Result cpu_run(Word *a, size_t n, double seconds, int threads, bool random,
                      const std::vector<int> &ids) {
    std::mutex mutex;
    std::condition_variable cv;
    int ready = 0;
    bool start = false;
    Clock::time_point begin, end;
    std::vector<uint64_t> counts(threads);
    std::vector<Clock::time_point> ends(threads);
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) workers.emplace_back([&, t] {
        cpu_set_t mask;
        CPU_ZERO(&mask); CPU_SET(ids[t], &mask);
        if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
            fail("could not pin a CPU worker");
        const size_t length = n / threads, bitmask = length - 1;
        Word *base = a + t * length;
        volatile Word *p = base;
        std::memset(base, 0, length * sizeof(Word)); // first touch outside timing
        size_t position = 0;
        uint32_t seed = uint32_t(t + 1);
        // Same worker and same access pattern for warmup and measurement.
        const auto warm_end = Clock::now() + std::chrono::milliseconds(200);
        do {
            for (int k = 0; k < 4096; ++k) {
                size_t i = random ? random_next(seed) & bitmask : position;
                p[i]++;
                position = (position + 1) & bitmask;
            }
        } while (Clock::now() < warm_end);
        std::memset(base, 0, length * sizeof(Word));
        position = 0; seed = uint32_t(t + 1);
        {
            std::unique_lock<std::mutex> lock(mutex);
            ++ready; cv.notify_all();
            cv.wait(lock, [&] { return start; });
        }
        uint64_t count = 0;
        do {
            for (int k = 0; k < 4096; ++k) {
                size_t i = random ? random_next(seed) & bitmask : position;
                p[i]++;
                position = (position + 1) & bitmask;
            }
            count += 4096;
        } while (Clock::now() < end);
        ends[t] = Clock::now();
        counts[t] = count;
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return ready == threads; });
        begin = Clock::now();
        end = begin + std::chrono::duration_cast<Clock::duration>(
                          std::chrono::duration<double>(seconds));
        start = true;
    }
    cv.notify_all();
    for (auto &worker : workers) worker.join();
    uint64_t count = 0;
    auto last = begin;
    for (int t = 0; t < threads; ++t) {
        count += counts[t];
        if (ends[t] > last) last = ends[t];
    }
    return {count, std::chrono::duration<double>(last - begin).count(), 0};
}

#ifndef CPU_PRIVATE_ONLY
__global__ void gpu_zero(Word *a, uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) a[i] = 0;
}
// Ordinary read/add/write, identical instructions for shared/private allocations.
// asm volatile prevents repeated increments becoming one load/add-N/store.
// .cg retains L2 caching where the allocation permits it, bypassing GPU L1.
__device__ __forceinline__ void increment(Word *p) {
    Word value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    ++value;
    asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(p), "r"(value) : "memory");
}
template<bool Random>
__global__ void gpu_updates(Word *a, uint32_t n, uint32_t passes) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    const uint32_t slots = n / stride, mask = slots - 1;
    uint32_t seed = tid + 1;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        for (uint32_t k = 0; k < slots; ++k) {
            uint32_t slot = k;
            if (Random) {
                seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                slot = seed & mask;
            }
            increment(a + tid + slot * stride); // each word has only one owner
        }
    }
}
static Result gpu_run(Word *a, uint32_t n, double seconds, int blocks, bool random) {
    cudaEvent_t first, last;
    CUDA(cudaEventCreate(&first));
    CUDA(cudaEventCreateWithFlags(&last, cudaEventBlockingSync));
    auto batch = [&](uint32_t passes) {
        CUDA(cudaEventRecord(first));
        if (random) gpu_updates<true><<<blocks, 128>>>(a, n, passes);
        else gpu_updates<false><<<blocks, 128>>>(a, n, passes);
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(last));
        CUDA(cudaEventSynchronize(last)); // host sleeps, never polls this buffer
        float ms;
        CUDA(cudaEventElapsedTime(&ms, first, last));
        return double(ms);
    };
    gpu_zero<<<blocks, 128>>>(a, n);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    uint32_t passes = 1;
    // Calibrate batches to ~20 ms so launch overhead is small and timing bounded.
    for (int k = 0; k < 12; ++k) {
        double ms = batch(passes);
        if (ms >= 10.0 || passes == (1U << 20)) break;
        double scale = ms > 0 ? 20.0 / ms : 4.0;
        if (scale > 4.0) scale = 4.0;
        uint32_t next = uint32_t(passes * scale);
        passes = next > passes ? next : passes + 1;
        if (passes > (1U << 20)) passes = 1U << 20;
    }
    gpu_zero<<<blocks, 128>>>(a, n); // discard all warmup/calibration increments
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    const auto begin = Clock::now();
    double active_ms = 0, elapsed = 0;
    uint64_t count = 0;
    do {
        const double ms = batch(passes);
        active_ms += ms;
        count += uint64_t(n) * passes;
        elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
        // Adapt for clock changes, always using the identical kernel for both cases.
        double scale = ms > 0 ? 20.0 / ms : 2.0;
        if (scale < 0.5) scale = 0.5;
        if (scale > 2.0) scale = 2.0;
        double next = passes * scale;
        if (next < 1) next = 1;
        if (next > (1U << 20)) next = 1U << 20;
        passes = uint32_t(next);
    } while (elapsed < seconds);
    CUDA(cudaEventDestroy(first)); CUDA(cudaEventDestroy(last));
    return {count, elapsed, active_ms / 1000.0};
}
#endif

int main(int argc, char **argv) {
    if (argc > 6 || (argc > 1 && !std::strcmp(argv[1], "--help"))) {
        std::printf("Usage: %s [MiB=64] [seconds=10] [CPU_threads=4] "
                    "[sequential|random=sequential] [repeats=1]\n", argv[0]);
        return argc > 6 ? 2 : 0;
    }
    auto number = [](const char *s) {
        char *tail;
        double value = std::strtod(s, &tail);
        if (tail == s || *tail || !std::isfinite(value)) fail("invalid numeric argument");
        return value;
    };
    const double mib = argc > 1 ? number(argv[1]) : 64;
    const double seconds = argc > 2 ? number(argv[2]) : 10;
    const double threads_arg = argc > 3 ? number(argv[3]) : 4;
    const bool random = argc > 4 && !std::strcmp(argv[4], "random");
    const double repeats_arg = argc > 5 ? number(argv[5]) : 1;
    if (mib < 0.0625 || mib > 4096 || seconds <= 0 || seconds > 3600 ||
        threads_arg < 1 || threads_arg > 8 || std::floor(threads_arg) != threads_arg ||
        repeats_arg < 1 || repeats_arg > 100 || std::floor(repeats_arg) != repeats_arg ||
        (argc > 4 && !random && std::strcmp(argv[4], "sequential"))) fail("invalid options");
    const size_t bytes = size_t(mib * 1048576);
    const size_t n = bytes / sizeof(Word);
    const int threads = int(threads_arg), repeats = int(repeats_arg);
    if (double(bytes) != mib * 1048576 || !power2(bytes) || !power2(threads))
        fail("MiB and CPU_threads must be powers of two (minimum size: 0.0625 MiB)");
    const auto ids = cpu_ids();
    if (size_t(threads) > ids.size()) fail("not enough CPUs in this process's affinity mask");
    int blocks = 0;
#ifndef CPU_PRIVATE_ONLY
    CUDA(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync));
    cudaDeviceProp prop{};
    CUDA(cudaGetDeviceProperties(&prop, 0));
    if (!prop.canMapHostMemory || prop.major != 7 || prop.minor != 2)
        fail("this build targets Xavier SM 7.2 with mapped host memory");
    blocks = 4 * prop.multiProcessorCount;
    if (!power2(blocks) || n < size_t(blocks * 128))
        fail("buffer/grid needs a power-of-two number of elements per GPU thread");
    std::fprintf(stderr, "%s, SM %d.%d, L2=%d bytes; GPU=%d blocks x 128 threads\n",
                 prop.name, prop.major, prop.minor, prop.l2CacheSize, blocks);
#else
    std::fprintf(stderr, "CPU_PRIVATE_ONLY validation build: no shared/GPU measurements\n");
#endif
    std::fprintf(stderr, "CPU workers pinned to:");
    for (int t = 0; t < threads; ++t) std::fprintf(stderr, " %d", ids[t]);
    std::fprintf(stderr, "; %.6g MiB; %s; %.3f seconds per case\n", mib,
                 random ? "random" : "sequential", seconds);
    std::puts("repeat,side,allocation,MiB,pattern,workers,updates,wall_seconds,"
              "Mupdates_per_s,GPU_active_seconds,GPU_active_Mupdates_per_s,checksum");
    for (int repeat = 1; repeat <= repeats; ++repeat) {
#ifndef CPU_PRIVATE_ONLY
        const int sides = 2, cases = 2;
#else
        const int sides = 1, cases = 1;
#endif
        for (int side = 0; side < sides; ++side) {
            double rates[2] = {0, 0};
            for (int c = 0; c < cases; ++c) {
                const bool shared = cases == 2 && ((repeat & 1) ? c == 1 : c == 0);
                Word *host = nullptr;
#ifndef CPU_PRIVATE_ONLY
                Word *device = nullptr;
                if (shared) {
                    // EXACT same allocation API and flags as atomic/ordering tests.
                    CUDA(cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocMapped));
                    CUDA(cudaHostGetDevicePointer(reinterpret_cast<void **>(&device), host, 0));
                } else if (side == 1) {
                    CUDA(cudaMalloc(reinterpret_cast<void **>(&device), bytes));
                } else
#endif
                {
                    if (posix_memalign(reinterpret_cast<void **>(&host), 4096, bytes))
                        fail("posix_memalign failed");
                }
                std::fprintf(stderr, "repeat %d: %s %s\n", repeat,
                             side ? "GPU" : "CPU", shared ? "shared" : "private");
                Result result{};
                if (side == 0) result = cpu_run(host, n, seconds, threads, random, ids);
#ifndef CPU_PRIVATE_ONLY
                else result = gpu_run(device, uint32_t(n), seconds, blocks, random);
                CUDA(cudaDeviceSynchronize()); // outside the timed interval
                std::vector<Word> copy;
                if (side == 1 && !shared) {
                    copy.resize(n);
                    CUDA(cudaMemcpy(copy.data(), device, bytes, cudaMemcpyDeviceToHost));
                    host = copy.data();
                }
#endif
                const uint64_t got = checksum(host, n);
                if (got != result.ops) fail("checksum mismatch (or a 32-bit element wrapped)");
                const double rate = result.ops / result.seconds / 1e6;
                rates[shared ? 1 : 0] = rate;
                std::printf("%d,%s,%s,%.6g,%s,%d,%llu,%.6f,%.6f,%.6f,%.6f,OK\n",
                            repeat, side ? "GPU" : "CPU", shared ? "shared" : "private",
                            mib, random ? "random" : "sequential", side ? blocks * 128 : threads,
                            static_cast<unsigned long long>(result.ops), result.seconds, rate,
                            result.gpu_seconds,
                            result.gpu_seconds ? result.ops / result.gpu_seconds / 1e6 : 0);
                std::fflush(stdout);
#ifndef CPU_PRIVATE_ONLY
                if (shared) CUDA(cudaFreeHost(host));
                else if (side == 1) CUDA(cudaFree(device));
                else
#endif
                    std::free(host);
            }
            if (cases == 2) std::fprintf(stderr,
                "%s shared/private=%.6f, shared throughput change=%+.3f%%\n",
                side ? "GPU" : "CPU", rates[1] / rates[0], 100 * (rates[1] / rates[0] - 1));
        }
    }
}
