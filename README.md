# K-Means algorithm:
This K-Means algorithm is based on Llyods algorithm which have two steps, Assignment and Update. Both the step are implemented for Nvidia GPUs (in CUDA).

## Includes
* K-Means wrapper for calculating the cluster allocation for all data points and also the cluster centers.
* File reader wrapper which enables read data specific to them.

## Usage
include *kmeans.h* header file into your program. It contains all the necessary header files and function definitions. <br>
**File reader wrapper:**
```
float** read_file(  char *filename,                //in  : data filename
                    size_t *nObjects,               //out : no. of datapoints
                    size_t *nFeatures)              //out : no. of features
//returns pointer to data objects of size [nObjectsPerRank][nFeatures]
```
File format for reading data:
Function reads ASCII encoded text file. The first line should be the number of data-points and number of features respectively in comma separated format. From next line, there should be individual data-point with comma separated features.
<br>

**K-Means wrapper**
```
float kmeans(   float **objects,                    //in  : objects pointer [nObjectsPerRank][nFeatures]
                size_t nObjects,                    //in  : no. of datapoints
                size_t nFeatures,                   //in  : no. of features
                float ***clusters,                  //out : cluster centers [nClusters][nFeatures]
                size_t nClusters,                   //in  : no. of clusters
                int **membership,                   //out : objects allocated clusters
                float threshold,                    //in  : threshold value for breaking algorithm
                int *nIteration)                    //out : no. of iteration run by kmeans
// returns total time taken by kmeans in ms.
```
**Notes**
* Threshold value is the percentage of overall data-points that changes the clusters in an iteration. ```Be careful with the threshold value``` as the algorithm can take a lot of time to converge to that threshold if datapoints are hard to separate.

## Data-point and feature size limitations
K-Means GPU kernel are designed to work on all data-point at a time in single iteration. If exceeding beyond that point, GPU will throw memory errors. Hence, care is been taken on how much data is being assigned to single compute node.
### **K-Means data-size limitation**
Max. number of data-points per compute node (rows) : 2,000,000 <br>
Max. number of features (columns) : 1024 <br>
 conditional on GPU memory size <br>

**Note**<br>
There is strict limitation of feature size to be at max. 1024, however, number of data-point can be further increased by little improvisation in the algorithm. Instead of processing all the data at a time, data can be processed in batches on the GPU, but this solution will cost in terms of performance.



## Sample Program
```
#include "kmeans.h"

int main() {

    char *filename = "data.txt";
    float **objects, **clusters;
    int *membership;
    int nIterations, ncluster=3;
    float threshold = 0.01;
    size_t nObjects, nFeatures;

    objects = read_file(filename, &nObjects, &nFeatures, rank, nranks);
    if(objects == NULL)
        return -1;

    float time = kmeans(objects, nObjects, nFeatures, &clusters, nCluster, &membership, threshold, &nIterations, rank, nranks);

    if(time==-1)
        return -1;


    free(objects[0]);
    free(clusters[0]);
    free(objects);
    free(clusters);
    free(membership);

}
```
