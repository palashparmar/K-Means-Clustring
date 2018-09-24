//cuda function declearation for main program

extern float* loadObjToDevice(float **objects,
                              size_t N,
                              size_t M);

extern float* allocateDistToDevice(size_t N,
                                   size_t M);

extern float* allocateClustersToDevice(float **clusters,
                                       size_t K,
                                       size_t M);

extern int* allocateMembershipToDevice(int *membership,
                                       size_t N);

extern float* allocateNewClusterToDevice(size_t K,
                                         size_t M);

extern int* allocateNewClusterSizeToDevice(size_t K);


extern int* allocateDelta();

extern void launch_distance_kernal(size_t N,
                                   size_t M,
                                   size_t K,
                                   float *device_objects,
                                   float *device_dist,
                                   float *device_clusters,
                                   int *device_membership,
                                   float *device_new_clusters,
                                   int *device_new_clusters_size,
                                   int *device_delta);

extern void launch_cluster_update_kernal(size_t K,
                                         size_t M,
                                         float *clusters,
                                         float *new_clusters,
                                         int *new_clusters_size);
