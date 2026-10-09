NVCC ?= /usr/local/cuda-11.4/bin/nvcc
CC ?= cc
CXX ?= g++
MEMORY_ARGS ?= 64 10 4 sequential 1

all: ordering_cpu_to_gpu ordering_gpu_to_cpu memory_cost

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

memory_cost: memory_cost.cu
	$(NVCC) -O2 -std=c++11 -arch=sm_72 -Xcompiler -pthread $< -o $@

run-memory: memory_cost
	./memory_cost $(MEMORY_ARGS)

# Two working sets, four cases per size, 10 seconds per case (about 80 s total).
run-memory-sweep: memory_cost
	sh ./run-memory-cost.sh

# Optional CPU-private-only build; does not substitute for the CUDA comparisons.
memory_cost_cpu_private: memory_cost.cu
	$(CXX) -x c++ -O2 -std=c++11 -pthread -DCPU_PRIVATE_ONLY $< -o $@

clean:
	rm -f atomic_test atomic_test_no_nvcc ordering_cpu_to_gpu ordering_gpu_to_cpu memory_cost memory_cost_cpu_private

.PHONY: all run run-no-nvcc run-ordering-cpu-to-gpu run-ordering-gpu-to-cpu run-ordering run-memory run-memory-sweep clean
