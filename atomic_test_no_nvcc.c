// Build: cc -O2 -std=c11 atomic_test_no_nvcc.c -ldl -o atomic_test_no_nvcc
// Requires the NVIDIA CUDA driver (libcuda.so.1), but no nvcc or CUDA headers.
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef uint64_t CUdeviceptr;
static int (*cuInit)(unsigned);
static int (*cuDeviceGet)(int *, int);
static int (*cuDeviceComputeCapability)(int *, int *, int);
static int (*cuCtxCreate_v2)(CUcontext *, unsigned, int);
static int (*cuMemHostAlloc)(void **, size_t, unsigned);
static int (*cuMemHostGetDevicePointer_v2)(CUdeviceptr *, void *, unsigned);
static int (*cuModuleLoadData)(CUmodule *, const void *);
static int (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static int (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                            unsigned, unsigned, unsigned, unsigned, void *,
                            void **, void **);
static int (*cuCtxSynchronize)(void);

#define LOAD(name) do { \
    *(void **)(&name) = dlsym(driver, #name); \
    if (!name) { fprintf(stderr, "Missing driver symbol: %s\n", #name); return 2; } \
} while (0)
#define CHECK(call) do { \
    int rc = (call); \
    if (rc) { fprintf(stderr, "%s failed (CUDA error %d)\n", #call, rc); return 2; } \
} while (0)

// The installed driver JIT-compiles this PTX for Xavier (SM 7.2).
static const char ptx[] =
    ".version 6.3\n"
    ".target sm_72\n"
    ".address_size 64\n"
    ".visible .entry gpu_add(.param .u64 arg) {\n"
    "  .reg .pred %p<1>; .reg .b32 %r<3>; .reg .b64 %rd<3>;\n"
    "  ld.param.u64 %rd0, [arg];\n"
    "  mov.u32 %r0, %tid.x;\n"
    "  setp.ne.u32 %p0, %r0, 0;\n"
    "  @%p0 bra poll;\n"
    "  add.u64 %rd1, %rd0, 4;\n"
    "  atom.global.sys.exch.b32 %r1, [%rd1], 1;\n"
    "poll:\n"
    "  add.u64 %rd1, %rd0, 8;\n"
    "  atom.global.sys.add.u32 %r1, [%rd1], 0;\n"
    "  setp.eq.u32 %p0, %r1, 0;\n"
    "  @%p0 bra poll;\n"
    "  mov.u32 %r0, 0;\n"
    "loop:\n"
    "  atom.global.sys.add.u32 %r1, [%rd0], 1;\n"
    "  add.u32 %r0, %r0, 1;\n"
    "  setp.lt.u32 %p0, %r0, 3125;\n"
    "  @%p0 bra loop;\n"
    "  ret;\n"
    "}\n";

struct shared { uint32_t counter, ready, start; };

int main(void) {
    void *driver = dlopen("libcuda.so.1", RTLD_NOW);
    if (!driver) { fprintf(stderr, "No NVIDIA CUDA driver: %s\n", dlerror()); return 2; }
    LOAD(cuInit); LOAD(cuDeviceGet); LOAD(cuDeviceComputeCapability);
    LOAD(cuCtxCreate_v2); LOAD(cuMemHostAlloc);
    LOAD(cuMemHostGetDevicePointer_v2); LOAD(cuModuleLoadData);
    LOAD(cuModuleGetFunction); LOAD(cuLaunchKernel); LOAD(cuCtxSynchronize);

    int dev, major, minor;
    CUcontext ctx;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDeviceComputeCapability(&major, &minor, dev));
    printf("GPU compute capability: %d.%d\n", major, minor);
    if (major != 7 || minor != 2) {
        fprintf(stderr, "SKIP: this PTX test targets Xavier SM 7.2\n");
        return 2;
    }
    CHECK(cuCtxCreate_v2(&ctx, 8, dev)); // 8 = CU_CTX_MAP_HOST

    struct shared *host;
    CUdeviceptr gpu_ptr;
    CUmodule module;
    CUfunction kernel;
    CHECK(cuMemHostAlloc((void **)&host, sizeof(*host), 2)); // 2 = DEVICEMAP
    host->counter = host->ready = host->start = 0;
    CHECK(cuMemHostGetDevicePointer_v2(&gpu_ptr, host, 0));
    CHECK(cuModuleLoadData(&module, ptx));
    CHECK(cuModuleGetFunction(&kernel, module, "gpu_add"));
    void *args[] = { &gpu_ptr };
    CHECK(cuLaunchKernel(kernel, 1, 1, 1, 64, 1, 1, 0, NULL, args, NULL));

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    time_t deadline = now.tv_sec + 5;
    while (__atomic_load_n(&host->ready, __ATOMIC_ACQUIRE) == 0) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec >= deadline) {
            fprintf(stderr, "FAIL: GPU did not signal ready in 5 seconds\n");
            _Exit(2);
        }
    }
    __atomic_store_n(&host->start, 1, __ATOMIC_RELEASE);
    for (unsigned i = 0; i < 200000; ++i)
        __atomic_fetch_add(&host->counter, 1, __ATOMIC_RELAXED);
    CHECK(cuCtxSynchronize());
    unsigned got = __atomic_load_n(&host->counter, __ATOMIC_RELAXED);
    printf("CPU + GPU: %u (expected 400000) — %s\n",
           got, got == 400000 ? "PASS" : "FAIL");
    return got == 400000 ? 0 : 1;
}
