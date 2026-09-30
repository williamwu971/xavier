NVCC ?= /usr/local/cuda-11.4/bin/nvcc
CC ?= cc

all: ordering_cpu_to_gpu ordering_gpu_to_cpu

atomic_test_no_nvcc: atomic_test_no_nvcc.c
	$(CC) -O2 -std=c11 -Wall -Wextra $< -ldl -o $@

run-no-nvcc: atomic_test_no_nvcc
	timeout 30s ./atomic_test_no_nvcc

atomic_test: atomic_test.cu
	$(NVCC) -O2 -std=c++11 -arch=sm_72 -Xcompiler -pthread $< -o $@

run: atomic_test
	timeout 30s ./atomic_test

ordering_cpu_to_gpu: ordering_test.cu
	$(NVCC) -O2 -std=c++11 -arch=sm_72 -Xcompiler -pthread -DCPU_TO_GPU $< -o $@

ordering_gpu_to_cpu: ordering_test.cu
	$(NVCC) -O2 -std=c++11 -arch=sm_72 -Xcompiler -pthread -DGPU_TO_CPU $< -o $@

run-ordering-cpu-to-gpu: ordering_cpu_to_gpu
	timeout 30s ./ordering_cpu_to_gpu

run-ordering-gpu-to-cpu: ordering_gpu_to_cpu
	timeout 30s ./ordering_gpu_to_cpu

run-ordering:
	$(MAKE) run-ordering-cpu-to-gpu
	$(MAKE) run-ordering-gpu-to-cpu

clean:
	rm -f atomic_test atomic_test_no_nvcc ordering_cpu_to_gpu ordering_gpu_to_cpu

.PHONY: all run run-no-nvcc run-ordering-cpu-to-gpu run-ordering-gpu-to-cpu run-ordering clean
