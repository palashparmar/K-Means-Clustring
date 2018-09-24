#include "kmeans.h"


float** read_file(  char *filename,
                    size_t *nObjects,
                    size_t *nFeatures)
{
    FILE *fp;
    size_t i,j;
    char line[3000], *token;
    char s[2] = ",";

    //opening file for reeding
    if((fp = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        return NULL;
    }

    //reading first line for nrows and ncols
    if((fgets(line, sizeof(line), fp)) == NULL) {
        fprintf(stderr, "Error: unable to read file (%s)\n", filename);
        return NULL;
    }

    //storing nrows and ncols
    *nObjects = atoi(strtok(line, s));
    *nFeatures = atoi(strtok(NULL, s));


    size_t N=*nObjects, M=*nFeatures;


    //storing size of objects for individual rank
    float **objects;

    //allocating memory for data-points
    objects = (float **)malloc(N*sizeof(float *));
    objects[0] = (float *)malloc(N*M*sizeof(float));
    for(i=0; i<N; i++)
        objects[i] = *objects + i*M;


    //reading data-points specific to current rank
    for(i=0; i<N; i++) {
        fgets(line, sizeof(line), fp);
        token = strtok(line, s);
        for(j=0; j<M; j++) {
            objects[i][j] = atof(token);
            token = strtok(NULL, s);

        }
    }

    //closing opened file pointer
    fclose(fp);

    return objects;

}
