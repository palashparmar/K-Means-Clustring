#include "kmeans.h"
#include <time.h>
#define THRESHOLD 0.001
#define N_CLUSTERS 3


int main(int argc, char *argv[])
{

    char *filename = "iris-data.txt";                             //datafile name
    float **objects, **clusters;
    int *membership;
    int loop_iteration;
    size_t nObjects, nFeatures, nObjectsPerRank;
    int i;

    objects = read_file(filename, &nObjects, &nFeatures);         //reading file

    if(objects == NULL)
    {
        return -1;
    }

    //calling kmeans
    float time = kmeans(objects, nObjects, nFeatures, &clusters, N_CLUSTERS, &membership, THRESHOLD, &loop_iteration);


    if(time == -1)
    {
        return -1;
    }

    //releasing memory
    free(objects[0]);
    free(objects);
    free(clusters[0]);
    free(clusters);
    free(membership);

    return 0;
}
