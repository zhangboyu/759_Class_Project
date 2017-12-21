# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -O3
ARCH   	:= -march=native
GENCODE_FLAGS := -gencode arch=compute_60,code=sm_60

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++11

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS :=
BIN = "/usr/local/gcc/6.4.0/bin/gcc"
.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : collision_detection

collision_detection: main.cu collision_gpu_broad.cu collision_gpu_narrow.cu collision_cpu.cu
	# module load cuda;nvcc -o collision_detection $(CXXSTD) $(GENCODE_FLAGS) $(OPT) main.cu collision_gpu_broad.cu collision_gpu_narrow.cu collision_cpu.cu -ccbin $(BIN)
	nvcc -o collision_detection $(CXXSTD) $(GENCODE_FLAGS) $(OPT) main.cu collision_gpu_broad.cu collision_gpu_narrow.cu collision_cpu.cu

.PHONY: clean
clean:
	@ rm -f *.gch collision_detection collision_pairs.txt
