PROJECT_NAME = kmeans_main

NVCC = nvcc
CC = gcc



BUILD_DIR = build

CFLAGS = -c -g
LFLAGS = -m64  -g -lm
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
	./$(BUILD_DIR)/$(PROJECT_NAME) 
