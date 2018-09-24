PROJECT_NAME = kmeans_main

NVCC = nvcc
CC = gcc


MPI_PATH = /opt/cray/pe/mpt/7.7.0.5/gni/mpich-gnu/5.1
CUDA_PATH = /opt/nvidia/cudatoolkit9.1/9.1.85_3.18-6.0.7.0_5.1__g2eb7c52
BUILD_DIR = build

CFLAGS = -c -g -I$(MPI_PATH)/include -I$(CUDA_PATH)/include -m64
LFLAGS = -m64  -g -L$(MPI_PATH)/lib -lmpich -lm
NVCCFLAGS = -m64 -g -c --gpu-architecture=sm_60


all: build clean


build: build_dir gpu cpu
	$(NVCC) $(LFLAGS) -o $(BUILD_DIR)/$(PROJECT_NAME) *.o


build_dir:
	mkdir -p $(BUILD_DIR)

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

cpu: kmeans file_io
	$(CC) $(CFLAGS) kmeans_main.c

kmeans:
	$(CC) $(CFLAGS) kmeans.c

file_io:
	$(CC) $(CFLAGS) file_io.c

clean:
	rm *.o

run:
	aprun ./$(BUILD_DIR)/$(PROJECT_NAME) iris_bdas.txt
