#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(CPU_TO_GPU) == defined(GPU_TO_CPU)
#error Define exactly one direction
#endif

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); \
        return 2; \
    } \
} while (0)

// This is a hardware litmus: ordinary volatile accesses, no atomic operations.
// A separate cache line holds each word; exactly one writer owns each pair.
struct alignas(128) Pair {
    alignas(64) volatile unsigned int a;
    alignas(64) volatile unsigned int b;
};
struct alignas(64) Stats {
    volatile unsigned long long ops;
    volatile unsigned long long nonzero;
    volatile unsigned long long bad;
};
struct Shared {
    Pair pair[4];
    Stats cpu[4];
    Stats gpu[128];
    alignas(64) volatile unsigned int ready;
    alignas(64) volatile unsigned int start;
    alignas(64) volatile unsigned int stop;
};

__global__ void gpu_work(Shared *s) {
    const int tid = threadIdx.x;
    if (tid == 0) s->ready = 1;
    __syncthreads();
    while (!s->start) { }

    volatile Pair *p = &s->pair[tid & 3];
    unsigned long long ops = 0, nonzero = 0, bad = 0;
    while (!s->stop) {
        for (int i = 0; i < 64; ++i) {
#ifdef CPU_TO_GPU
            const unsigned int b = p->b;
            __threadfence_system();
            const unsigned int a = p->a;
            nonzero += b != 0;
            bad += a < b;
#else
            p->a++;
            // __threadfence_system();
            p->b++;
#endif
        }
        ops += 64;
        s->gpu[tid].ops = ops;
        s->gpu[tid].nonzero = nonzero;
        s->gpu[tid].bad = bad;
    }
}

static unsigned long long total_ops(const Stats *stats, int n) {
    unsigned long long total = 0;
    for (int i = 0; i < n; ++i) total += stats[i].ops;
    return total;
}
static unsigned long long total_nonzero(const Stats *stats, int n) {
    unsigned long long total = 0;
    for (int i = 0; i < n; ++i) total += stats[i].nonzero;
    return total;
}
static unsigned long long total_bad(const Stats *stats, int n) {
    unsigned long long total = 0;
    for (int i = 0; i < n; ++i) total += stats[i].bad;
    return total;
}

int main() {
    CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("%s, %s (SM %d.%d)\n",
#ifdef CPU_TO_GPU
                "CPU writes / GPU reads",
#else
                "GPU writes / CPU reads",
#endif
                prop.name, prop.major, prop.minor);
    if (prop.major != 7 || prop.minor != 2 || !prop.canMapHostMemory) {
        std::fprintf(stderr, "SKIP: built for Xavier SM 7.2 + mapped host memory\n");
        return 2;
    }

    Shared *host, *device;
    CHECK(cudaHostAlloc(reinterpret_cast<void **>(&host), sizeof(*host),
                        cudaHostAllocMapped));
    std::memset(host, 0, sizeof(*host));
    CHECK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&device), host, 0));
#ifdef CPU_TO_GPU
    gpu_work<<<1, 128>>>(device); // 32 readers for each CPU writer
#else
    gpu_work<<<1, 4>>>(device);   // one GPU writer for each CPU reader
#endif
    CHECK(cudaGetLastError());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!host->ready) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::fprintf(stderr, "FAIL: GPU did not signal ready in 5 seconds\n");
            std::_Exit(2);
        }
    }

    std::thread cpu[4];
    for (int lane = 0; lane < 4; ++lane) {
        cpu[lane] = std::thread([host, lane] {
            volatile Pair *p = &host->pair[lane];
            unsigned long long ops = 0, nonzero = 0, bad = 0;
            while (!host->start) { }
            while (!host->stop) {
                for (int i = 0; i < 64; ++i) {
#ifdef CPU_TO_GPU
                    p->a++;
                    __sync_synchronize();
                    p->b++;
#else
                    const unsigned int b = p->b;
                    __sync_synchronize();
                    const unsigned int a = p->a;
                    nonzero += b != 0;
                    bad += a < b;
#endif
                }
                ops += 64;
                host->cpu[lane].ops = ops;
                host->cpu[lane].nonzero = nonzero;
                host->cpu[lane].bad = bad;
            }
        });
    }

    __sync_synchronize();
    host->start = 1;
    const auto begin = std::chrono::steady_clock::now();
    unsigned long long prev_writes = 0, prev_reads = 0;
    int overlapping = 0;
    for (int sec = 1; sec <= 10; ++sec) {
        std::this_thread::sleep_until(begin + std::chrono::seconds(sec));
#ifdef CPU_TO_GPU
        const auto writes = total_ops(host->cpu, 4);
        const auto reads = total_ops(host->gpu, 128);
#else
        const auto writes = total_ops(host->gpu, 4);
        const auto reads = total_ops(host->cpu, 4);
#endif
        const bool both = writes > prev_writes && reads > prev_reads;
        overlapping += both;
        std::printf("second %2d: writes +%llu, reads +%llu %s\n",
                    sec, writes - prev_writes, reads - prev_reads,
                    both ? "(both active)" : "");
        prev_writes = writes;
        prev_reads = reads;
    }
    host->stop = 1;
    __sync_synchronize();
    for (auto &t : cpu) t.join();
    CHECK(cudaDeviceSynchronize());

#ifdef CPU_TO_GPU
    const auto writes = total_ops(host->cpu, 4);
    const auto reads = total_ops(host->gpu, 128);
    const auto valid = total_nonzero(host->gpu, 128);
    const auto bad = total_bad(host->gpu, 128);
#else
    const auto writes = total_ops(host->gpu, 4);
    const auto reads = total_ops(host->cpu, 4);
    const auto valid = total_nonzero(host->cpu, 4);
    const auto bad = total_bad(host->cpu, 4);
#endif
    std::printf("writes=%llu, reads=%llu (b>0: %llu), a<b=%llu, overlap=%d/10\n",
                writes, reads, valid, bad, overlapping);
    bool final_ok = true;
    for (int lane = 0; lane < 4; ++lane) {
        const auto a = host->pair[lane].a, b = host->pair[lane].b;
        std::printf("pair %d: a=%u b=%u\n", lane, a, b);
        final_ok &= a == b;
    }
    const int status = bad || !final_ok ? 1 :
                       (overlapping < 8 || valid == 0 || writes == 0) ? 2 : 0;
    std::puts(status == 0 ? "PASS (observed)" : status == 1 ? "FAIL: ordering/count mismatch" :
              "INCONCLUSIVE: insufficient overlapping progress");
    CHECK(cudaFreeHost(host));
    return status;
}
