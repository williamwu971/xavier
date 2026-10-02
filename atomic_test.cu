#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); \
        return 2; \
    } \
} while (0)

struct Shared {
    alignas(64) unsigned long long counter;
    alignas(64) unsigned long long cpu_ops;
    alignas(64) unsigned long long gpu_ops;
    alignas(64) unsigned int ready;
    alignas(64) unsigned int start;
    alignas(64) unsigned int stop;
};

__global__ void gpu_add(Shared *s) {
    if (threadIdx.x == 0) atomicExch_system(&s->ready, 1);
    __syncthreads();
    while (atomicAdd_system(&s->start, 0) == 0) { }

    while (atomicAdd_system(&s->stop, 0) == 0) {
        for (int i = 0; i < 256; ++i)
            // s->counter++;
            atomicAdd_system(&s->counter, 1ULL);
        atomicAdd_system(&s->gpu_ops, 256ULL);
    }
}

int main() {
    CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (SM %d.%d), concurrentManagedAccess=%d\n",
                prop.name, prop.major, prop.minor, prop.concurrentManagedAccess);
    if (prop.major < 7 || (prop.major == 7 && prop.minor < 2) ||
        !prop.canMapHostMemory) {
        std::fprintf(stderr, "SKIP: needs SM 7.2+ and mapped host memory\n");
        return 2;
    }

    Shared *host, *device;
    CHECK(cudaHostAlloc(reinterpret_cast<void **>(&host), sizeof(*host),
                        cudaHostAllocMapped));
    host->counter = host->cpu_ops = host->gpu_ops = 0;
    host->ready = host->start = host->stop = 0;
    CHECK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&device), host, 0));

    gpu_add<<<1, 128>>>(device);
    CHECK(cudaGetLastError());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (__atomic_load_n(&host->ready, __ATOMIC_ACQUIRE) == 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::fprintf(stderr, "FAIL: GPU did not signal ready in 5 seconds\n");
            std::_Exit(2);
        }
    }

    std::thread workers[4];
    for (auto &worker : workers) {
        worker = std::thread([host] {
            while (__atomic_load_n(&host->start, __ATOMIC_ACQUIRE) == 0) { }
            while (__atomic_load_n(&host->stop, __ATOMIC_ACQUIRE) == 0) {
                for (int i = 0; i < 256; ++i)
                    __atomic_fetch_add(&host->counter, 1ULL, __ATOMIC_RELAXED);
                    // host->counter++;
                __atomic_fetch_add(&host->cpu_ops, 256ULL, __ATOMIC_RELAXED);
            }
        });
    }

    __atomic_store_n(&host->start, 1, __ATOMIC_RELEASE);
    const auto begin = std::chrono::steady_clock::now();
    unsigned long long prev_cpu = 0, prev_gpu = 0;
    int overlap = 0;
    for (int sec = 1; sec <= 10; ++sec) {
        std::this_thread::sleep_until(begin + std::chrono::seconds(sec));
        const auto cpu = __atomic_load_n(&host->cpu_ops, __ATOMIC_RELAXED);
        const auto gpu = __atomic_load_n(&host->gpu_ops, __ATOMIC_RELAXED);
        const bool both = cpu > prev_cpu && gpu > prev_gpu;
        overlap += both;
        std::printf("second %2d: CPU +%llu, GPU +%llu %s\n",
                    sec, cpu - prev_cpu, gpu - prev_gpu, both ? "(both active)" : "");
        prev_cpu = cpu;
        prev_gpu = gpu;
    }
    __atomic_store_n(&host->stop, 1, __ATOMIC_RELEASE);
    for (auto &worker : workers) worker.join();
    CHECK(cudaDeviceSynchronize());

    const auto cpu = __atomic_load_n(&host->cpu_ops, __ATOMIC_RELAXED);
    const auto gpu = __atomic_load_n(&host->gpu_ops, __ATOMIC_RELAXED);
    const auto got = __atomic_load_n(&host->counter, __ATOMIC_RELAXED);
    std::printf("counter=%llu, CPU=%llu + GPU=%llu = %llu; overlap=%d/10 seconds\n",
                got, cpu, gpu, cpu + gpu, overlap);
    const int status = got != cpu + gpu ? 1 : overlap < 8 ? 2 : 0;
    std::puts(status == 0 ? "PASS" : status == 1 ? "FAIL: lost updates" :
              "INCONCLUSIVE: insufficient measured overlap");
    CHECK(cudaFreeHost(host));
    return status;
}
