#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper.h"



//kernal function for calculating distance between a data-point and cluster
__device__ float euclidian_distance_gpu(float *object, float *cluster, float *dist, int M) {
    int y = threadIdx.y;
    //squaring the features for object
    if(y<M) {
        dist[y] = (object[y] - cluster[y])*(object[y] - cluster[y]);
    }

    __syncthreads();
    //reduced sum for addition of squared features
    size_t lIndex = M/2;
    size_t hIndex = ((M-1)/2)+1;

    while(lIndex>0) {
        if(y<lIndex) {
            dist[y] += dist[y+hIndex];
        }
        __syncthreads();
        lIndex = hIndex/2;
        hIndex = ((hIndex-1)/2)+1;
    }

    //sum in first index of array after reduced sum
    return dist[0];
}

//assignment step kernal in kmeans
__global__ void find_nearest_cluster(float *objects, float *clusters, float *dist, int *membership, size_t N, size_t M, size_t K, float *new_clusters, int *new_clusters_size, int *delta) {

    *delta = 0;                         //number of data-points changing clusters
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = threadIdx.y;



    if(x<N) {
        //calculating distance for each clusters and identifying cluster number with least distance
        float distance = euclidian_distance_gpu(&objects[x*M], &clusters[0], &dist[x*M], M);
        int n = 0;
        float new_distance;
        for(size_t i=1; i<K; i++) {
            new_distance = euclidian_distance_gpu(&objects[x*M], &clusters[i*M], &dist[x*M], M);
            if(distance > new_distance) {
                distance = new_distance;
                n = i;
            }
        }
        if(y==0) {
            //updating the membership array with new value and calculating delta's
            if(membership[x] != n) {
                atomicAdd(delta,1);
                membership[x] = n;
            }
        }

        //noting new clusters and their numbers for current ranks
        atomicAdd(&new_clusters[n*M+y],objects[x*M+y]);
        if(y==0)
            atomicAdd(&new_clusters_size[n],1);

    }
}

//updating step kernal in kmeans
__global__ void update_clusters(size_t K, size_t M, float *clusters, float *new_clusters, int *new_clusters_size) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = threadIdx.y;
    if(x<K) {
        if(y<M) {
            //updating clusters and resetting it for future use
            clusters[x*M+y] = new_clusters[x*M+y]/new_clusters_size[x];
            new_clusters[x*M+y] = 0.0;
            if(y==0)
                new_clusters_size[x] = 0;
        }
    }
}


//SOME HELPER FUNCTIONS FOR SETTING UP GPU

//allocating and loading data onto GPU
extern "C" float* loadObjToDevice(float **objects, size_t N, size_t M) {
    float *dev_object;
    gpuErrchk(cudaMalloc(&dev_object, N*M*sizeof(float)));
    gpuErrchk(cudaMemcpy(dev_object, &objects[0][0], N*M*sizeof(float), cudaMemcpyHostToDevice));
    return dev_object;

}

//allocating temp variable for distance calculation
extern "C" float* allocateDistToDevice(size_t M, size_t N) {
    float *dev_dist;
    gpuErrchk(cudaMalloc(&dev_dist, N*M*sizeof(float)));
    return dev_dist;
}

//allocaitng and loading initial clusters to device
extern "C" float* allocateClustersToDevice(float **clusters, size_t K, size_t M) {
    float *dev_clusters;
    gpuErrchk(cudaMalloc(&dev_clusters, K*M*sizeof(float)));
    gpuErrchk(cudaMemcpy(dev_clusters, &clusters[0][0], K*M*sizeof(float), cudaMemcpyHostToDevice));
    return dev_clusters;
}

//allocating membership array onto GPU
extern "C" int* allocateMembershipToDevice(int *membership, size_t N) {
    int *dev_membership;
    gpuErrchk(cudaMalloc(&dev_membership, N*sizeof(int)));
    gpuErrchk(cudaMemcpy(dev_membership, &membership[0], N*sizeof(int), cudaMemcpyHostToDevice));
    return dev_membership;
}

//allocating extra memory for new cluster onto GPU
extern "C" float* allocateNewClusterToDevice(size_t K, size_t M) {
    float *dev_new_clusters;
    gpuErrchk(cudaMalloc(&dev_new_clusters, K*M*sizeof(float)));
    return dev_new_clusters;
}


extern "C" int* allocateNewClusterSizeToDevice(size_t K)
{
    int *dev_new_clusters_size;
    gpuErrchk(cudaMalloc(&dev_new_clusters_size, K*sizeof(int)));
    return dev_new_clusters_size;
}

//allocating delta for threshold calculation
extern "C" int* allocateDelta() {
    int *dev_delta;
    gpuErrchk(cudaMalloc(&dev_delta, sizeof(int)));
    return dev_delta;
}


//function for assignment call
extern "C" void launch_distance_kernal( size_t N, size_t M, size_t K, float *objects, float *dist, float *clusters, int *membership, float *new_clusters, int *new_clusters_size, int *delta) {
    //threads and blocks structure for GPU
    unsigned int thread_size = 1024/M;
    unsigned int block_size = (N +thread_size-1)/(thread_size);
    dim3 blockspergrid = {block_size,1};
    dim3 threadsperblocks = {thread_size, (unsigned int)M};
    //calling GPU kernal
    find_nearest_cluster<<<blockspergrid,threadsperblocks>>>(objects, clusters, dist, membership, N, M, K, new_clusters, new_clusters_size, delta, rank);
    //checking sync and errors
    cudaDeviceSynchronize();
    checkLastCudaError();
}

//function for update call
extern "C" void launch_cluster_update_kernal(size_t K, size_t M, float *clusters, float *new_clusters, int *new_clusters_size) {
    //threads and blocks structure on GPU
    unsigned int thread_size = 1024/M;
    unsigned int block_size = (K + thread_size - 1)/(thread_size);

    dim3 blockspergrid = {block_size, 1};
    dim3 threadsperblocks = {thread_size, (unsigned int)M};
    //calling kernal
    update_clusters<<<blockspergrid, threadsperblocks>>>(K, M, clusters, new_clusters, new_clusters_size, rank);
    //sync and error check
    cudaDeviceSynchronize();
    checkLastCudaError();
}
