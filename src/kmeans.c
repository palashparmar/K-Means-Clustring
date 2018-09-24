#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kmeans_cuda.h"


//added for performance measurement
double get_wall_time()
{
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec*0.000001;
}


//start from here
float kmeans(float **objects, size_t L, size_t M, float ***clusters, size_t K, int **membership, float threshold, int *loop_iteration) {

    size_t i=0, j=0, k=0;     //temp vars



//assign memory for clusters and temp clusters centers
    float **new_clusters;
    int *new_clusters_size;
    new_clusters_size = (int *)malloc(K*sizeof(int));
    for(i=0; i<K; i++)
        new_clusters_size[i] = 0;

    *clusters  = (float **)malloc(K*sizeof(float *));
    (*clusters)[0] = (float *)malloc(M*K*sizeof(float));
    new_clusters = (float **)malloc(K*sizeof(float *));
    new_clusters[0] = (float *)malloc(M*K*sizeof(float));

//assigning initial K datapoints as cluster centres
    for(i=0; i<K; i++) {
        (*clusters)[i] = (**clusters + i*M);
        new_clusters[i] = (*new_clusters + i*M);

        for(k=0; k<M; k++) {
            (*clusters)[i][k] = objects[i][k];
            new_clusters[i][k] = 0.0;
        }
    }


//allocating memory for data global and local membership array
    *membership = (int *)malloc(L*sizeof(int));


    int delta = L;     //parameter for stoping iterations
    *loop_iteration = 0;
    double tic, toc;     //start/end time of program

//start of time measurement
    tic = get_wall_time();
    int n;

//GPU memory allocation and uploading data
    float *device_objects = loadObjToDevice(&objects[0], L, M);
    float *device_dist = allocateDistToDevice(L, M);
    float *device_clusters = allocateClustersToDevice(*clusters, K, M);
    int *device_membership = allocateMembershipToDevice(local_membership, L);
    float *device_new_clusters = allocateNewClusterToDevice(K, M);
    int *device_new_clusters_size = allocateNewClusterSizeToDevice(K);
    int *device_delta = allocateDelta();


//gpu_memory();
//MPI_Barrier(MPI_COMM_WORLD);
//starting iterations
    while(((float)delta/L) > threshold) {
//while(*loop_iteration<2) {          //for testing and benchmarking, limiting number of iterations

//launching cuda kernel for calculating nearest cluster for all objects
        launch_distance_kernal(L, M, K, device_objects, device_dist, device_clusters, device_membership, device_new_clusters, device_new_clusters_size, device_delta);

//launching cuda kernel for updating cluster centers
        launch_cluster_update_kernal(K, M, device_clusters, device_new_clusters, device_new_clusters_size);
        *loop_iteration++;
    }


//getting membership and cluster information for GPU
    cudaMemcpy(membership, device_membership, L*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_clusters_size, device_new_clusters_size, K*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(clusters[0][0], device_clusters, K*M*sizeof(float), cudaMemcpyDeviceToHost);



//end of time measurement
    toc = get_wall_time();

//releasing memory on GPU
    cudaFree(device_objects);
    cudaFree(device_dist);
    cudaFree(device_clusters);
    cudaFree(device_membership);
    cudaFree(device_new_clusters);
    cudaFree(device_new_clusters_size);
    cudaFree(device_delta);


//releasing memory on CPU
    free(new_clusters);
    free(new_clusters_size);


    return toc-tic;
}
