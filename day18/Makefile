NVCC ?= nvcc

# If you want to target a specific GPU, set SM, e.g. `make SM=90`.
SM ?=
ifeq ($(SM),)
GENCODE :=
else
GENCODE := -arch=sm_$(SM)
endif

NVCCFLAGS ?= -O3 -std=c++17 -lineinfo

# Optional: enable CuTe DSL kernels from CUTLASS (header-only).
CUTLASS_DIR ?= $(HOME)/opensource/cutlass
CUTLASS_INC ?= $(CUTLASS_DIR)/include

ifneq ("$(wildcard $(CUTLASS_INC)/cute/algorithm/tuple_algorithms.hpp)","")
CUTE_INC := -I$(CUTLASS_INC)
else
CUTE_INC :=
endif

.PHONY: all clean

all: reduction interleaved_addressing

reduction: reduction.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $(CUTE_INC) $< -o $@

interleaved_addressing: interleaved_addressing.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $< -o $@

clean:
	rm -f reduction interleaved_addressing

