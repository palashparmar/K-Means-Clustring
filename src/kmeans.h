#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>

//returns time taken in kmeans computation
float kmeans(float **object,                //in  :  objects[nObjs][nFeatures]
             size_t nObjects,               //in  :  no. objects
             size_t nFeatures,              //in  :  no. features
             float ***clusters,             //out :  cluster center [nClusters][nFeatures]
             size_t nClusters,              //in  :  no. clusters
             int **membership,              //out :  obj. clusters
             float threshold,               //in  :  threshold
             int *loopIteration);           //out :  total iterations



//return pointer to data objects [nObjectsPerRank][nFeatures]
float** read_file(char *filename,               //in  :  data filename
                  size_t *nObjects,             //out :  no. objects
                  size_t *nFeatures);           //out :  no. features
