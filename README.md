# K-Means algorithm:
This K-Means algorithm is based on Llyods algorithm which have two steps, Assignment and Update. Both the step are implementated on GPU with a MPI Communication in between which communicate the current cluster centers to all other ranks. Since the communication step has GPU->CPU->CPU->GPU flow, It might *not be a good idea to run the algorithm for small dataset as performace gain would be much less than overall communication overhead*.

## Includes
* K-Means wrapper for calculating the cluster allocation for all datapoints and also the cluster centers.
* File reader wrapper which enables individual rank to read data specific to them.

## Usage
include *kmeans.h* header file into your program. It contains all the necessary header files and function definations. <br>
**File reader wrapper:**
```
float** read_file(  char *fileneame,                //in  : data filename
                    size_t *nObjects,               //out : no. of datapoints
                    size_t *nObjectsPerRank,        //out : no. of datapoints per rank
                    size_t *nFeatures,              //out : no. of features
                    int rank,                       //in  : current process rank
                    int nranks);                    //in  : size of mpi jobs (total ranks)
//returns pointer to data objects of size [nObjectsPerRank][nFeatures]
```
File format for reading data:
Function reads ASCII encoded text file. The first line should be the number of data-points and number of features respectively in comma separated format. From next line, there should be individual data-point with comma separated features.
<br>

**K-Means wrapper**
```
float kmeans(   float **objects,                    //in  : objects pointer [nObjectsPerRank][nFeatures]
                size_t nObjects,                    //in  : no. of datapoints
                size_t nObjectsPerRank,             //in  : no. of datapoints per rank
                size_t nFeatures,                   //in  : no. of features
                float ***clusters,                  //out : cluster centers [nClusters][nFeatures]
                size_t nClusters,                   //in  : no. of clusters
                int **membership,                   //out : objects allocated clusters
                float threshold,                    //in  : threshold value for breaking algorithm
                int *nIteration,                    //out : no. of iteration run by kmeans
                int rank,                           //in  : current process rank
                int nranks);                        //in  : size of mpi jobs (total ranks)
// returns total time taken by kmeans in ms.
```
**Notes**
* Before calling any of the functions, make sure the MPI initializations are done properly and also get the current rank and total size of MPI before hand. After done with overall computation, make sure to finalize MPI at the end of the program.
* Threshold value is the percentage of overall data-points that changes the clusters in an iteration. ```Be careful with the threshold value``` as the algorithm can take a lot of time to converge to that threshold if datapoints are hard to separate.

## Data-point and feature size limitations
Optimized Cray K-Means GPU kernel are designed to work on all data-point at a time in single iteration. If exceeding
beyond that point, GPU will throw memory errors. Hence, care is been taken on how much data is being assigned to sing
le compute node.
### **K-Means data-size limitation**
Max. number of data-points per compute node (rows) : 8,000,000 <br>
Max. number of features (columns) : 1024 <br>

**Note**<br>
There is strict limitation of feature size to be at max. 1024, however, number of data-point can be further increased by little improvisation in the algorithm. Instead of processing all the data at a time, data can be processed in batches on the GPU, but this solution will cost in terms of performance. **The best idea would be to increase the number of compute nodes to process more data at a time.**



## Sample Program
```
#include "kmeans.h"

int main() {

    char *filename = "data.txt";
    float **objects, **clusters;
    int *membership;
    int nIterations, ncluster=3;
    float threshold = 0.01;
    size_t nObjects, nObjectsPerRank, nFeatures;

    objects = read_file(filename, &nObjects, &nObjectsPerRank, &nFeatures, rank, nranks);
    if(objects == NULL)
        return -1;

    float time = kmeans(objects, nObjects, nObjectsPerRank, nFeatures, &clusters, nCluster, &membership, threshold, &nIterations, rank, nranks);

    if(time==-1)
        return -1;


    free(objects[0]);
    free(clusters[0]);
    free(objects);
    free(clusters);
    free(membership);

}
```
