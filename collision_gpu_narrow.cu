#include "collision_gpu_narrow.h"

// couple notes here:
// 1. there are two ways to explore the parallelization for this collision test task,
//    1). perform test simultaneously with many object pairs (one collision test per thread)
//    2). for each pair of objects that needs to be tested, in gjk algorithm, parallize the part that finds the farest point along a direction in objects
//        (one collision test per block)
//    Because we will use the vertices' coordinates info many times, so we want to bring the coordinates info into shared memory.
//    So, the first path has stricter limitation on how many vertices an object can have. Shared memory 64KB, each float 4B, 3 coordinates for each vertex (3D)
//    which means if we have 32 threads per block, and each thread handles a pair of object, then the number of vertices each object can have (on average) is:
//    64 * 1024 / 4 / 3 / 2 / 32 = 85. This a relative small number in terms of number of vertices of an object, one may say that this number is too small to
//    be any useful to present an object in real world, but we can always use convex decomposition to decompose an complicated object into many many small convex
//    objects which meet this requirement. So, the first path is OK
//    The second path can handle an object with 32 * 85 = 2730 vertices, the part canot be parallized is the part that tests if the origin is in the Minkowski
//    difference, but this part is not very long and can be easily executed by one thread. The good part of this path are: 1) when bring coordinates data from
//    global memory to shared memory, the access is coalesced; 2) when executing support mapping function, the access to shared memory has less bank conflict.
//    We will simply have as many blocks as the number of collision tests we want to perform.
//    We decided to go with the second path.
// 2. We need to be careful when bring data from global memory to shared memory, we want it to be coalesced.

__global__ void cuda_test_collision(uint32_t * d_first_obj_index,
                                    uint32_t * d_second_obj_index,
                                    uint32_t * d_num_vertices,
                                    uint32_t * d_obj_starting_index,
                                    float * d_vertices_coordinate_x,
                                    float * d_vertices_coordinate_y,
                                    float * d_vertices_coordinate_z,
                                    unsigned int dim,
                                    uint32_t d_max_num_vertices,
                                    bool * d_collision_results);

__device__ void bring_data_to_shared_mem(float * destination, float * source, uint32_t num, uint32_t starting_index);
__device__ void support_Minkowski_sum(float * first_obj_coordinate_x,
                                    float * first_obj_coordinate_y,
                                    float * first_obj_coordinate_z,
                                    float * second_obj_coordinate_x,
                                    float * second_obj_coordinate_y,
                                    float * second_obj_coordinate_z,
                                    uint32_t first_obj_num_vertices,
                                    uint32_t second_obj_num_vertices,
                                    float * direction_vector,
                                    float * dot_prod,
                                    unsigned int * max_index,
                                    float * temp_point_1,
                                    float * temp_point_2,
                                    float * point,
                                    unsigned int dim);
__device__ void support_one_object(float * coordiante_x,
                                    float * coordinate_y,
                                    float * coordinate_z,
                                    uint32_t num_vertices,
                                    float * direction_vector,
                                    float * dot_prod,
                                    unsigned int * max_index,
                                    float * point,
                                    unsigned int dim,
                                    bool negate);
__device__ void check_two_face(float * ab, float * ac, float * ad, float * ao, float * abc, float * acd,
                               float * local_a, float * local_b, float * local_c, float * local_d,
                               float * direction_vector, unsigned int * simplex_size, unsigned int dim);
__device__ void check_one_face(float * ab, float * ac, float * ao, float * abc,
                                float * local_a, float * local_b, float * local_c, float * local_d,
                                float * direction_vector, unsigned int * simplex_size, unsigned int dim);
__device__ void update(float * a, float * b, float * c, float * d, unsigned int * simplex_size, float * direction_vector, bool * result, bool * finish,
                       unsigned int dim);
__device__ void assign(float * left, float * right, unsigned int dim, bool negate);
__device__ void vector_subtract(float * result, float * head, float * tail, unsigned int dim);
__device__ void cross_product(float * result, float * a, float * b);
__device__ float dot_product(float * a, float * b);

void test_collision_gpu(unsigned long num_objects,
                        unsigned long num_collision,
                        uint32_t * d_first_obj_index,
                        uint32_t * d_second_obj_index,
                        uint32_t * d_num_vertices, // for all objects in the space
                        float * d_vertices_coordinate_x,// for all objects in the space
                        float * d_vertices_coordinate_y,// for all objects in the space
                        float * d_vertices_coordinate_z,// for all objects in the space
                        unsigned int dim,
                        bool * d_collision_results,
                        float * time) {

    // for each SM to bring coordinates data from global memory to shared memory, it needs to know the starting index in memory for the objects its needs to take
    // so we perform an exclusive scan on the d_num_vertices array.
    uint32_t * d_obj_starting_index;
    cudaMalloc((void **)&d_obj_starting_index, num_objects * sizeof(uint32_t));
    thrust::exclusive_scan(thrust::device, d_num_vertices, d_num_vertices + num_objects, d_obj_starting_index);
    // we need to find the maximum number of vertices of an object to calclulate the upper bound of the size of the shared memory of a block
    // the shared memory for a block needs to hold the following data:
    // 1). two objects vertices' coordinates;
    // 2). for each vertex, we need to have the space to store the dot product between it and the direction vector (this can be shared by the two objects)
    // 3). we need to have space to store the intermediate index values when looking for the index of the maximum value in the dot product array
    // 4). since support mapping function is executed by all threads but the checking for origin is done by only one thread, we need to have some communication
    //     between this one thread and all threads, we use shared mem to do this work. Specifically, three floats for direction vector, three floats for
    //     each of the point a, b, c, d and two temp points, one int for the simplex size, one bool to indicate when that one thread finishes,
    //     and one bool for final result.
    uint32_t d_max_num_vertices = thrust::reduce(thrust::device, d_num_vertices, d_num_vertices + num_objects, 0, thrust::maximum<uint32_t>());
    uint32_t d_shared_mem_size = (2 * dim * d_max_num_vertices + 2 * d_max_num_vertices + 7 * dim + 3) * sizeof(float);
    // like the previous kernels, we use 512 threads per block
    unsigned int thread_per_block = 512;

    cudaEvent_t startEvent_inc, stopEvent_inc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
	cudaEventRecord(startEvent_inc,0);
    cuda_test_collision<<<num_collision, thread_per_block, d_shared_mem_size>>>(d_first_obj_index,
                                                                                d_second_obj_index,
                                                                                d_num_vertices,
                                                                                d_obj_starting_index,
                                                                                d_vertices_coordinate_x,
                                                                                d_vertices_coordinate_y,
                                                                                d_vertices_coordinate_z,
                                                                                dim,
                                                                                d_max_num_vertices,
                                                                                d_collision_results);
    cudaEventRecord(stopEvent_inc,0);
    cudaEventSynchronize(stopEvent_inc);
    cudaEventElapsedTime(time, startEvent_inc, stopEvent_inc);
    cudaFree(d_obj_starting_index);
}

__global__ void cuda_test_collision(uint32_t * d_first_obj_index,
                                    uint32_t * d_second_obj_index,
                                    uint32_t * d_num_vertices,
                                    uint32_t * d_obj_starting_index,
                                    float * d_vertices_coordinate_x,
                                    float * d_vertices_coordinate_y,
                                    float * d_vertices_coordinate_z,
                                    unsigned int dim,
                                    uint32_t d_max_num_vertices,
                                    bool * d_collision_results){

    // setup the shared memory and properly partition it
    extern volatile __shared__ float shared_mem[];
    float * first_obj_coordinate_x, * first_obj_coordinate_y, * first_obj_coordinate_z,
          * second_obj_coordinate_x, * second_obj_coordinate_y, * second_obj_coordinate_z,
          * dot_prod, * direction_vector, * a, * b, * c, * d, * temp_point_1, * temp_point_2;
    unsigned int * max_index, * simplex_size;
    bool * finish, * result;
    first_obj_coordinate_x = (float *)&shared_mem[0];
    first_obj_coordinate_y = (float *)&shared_mem[d_max_num_vertices];
    second_obj_coordinate_x = (float *)&shared_mem[dim * d_max_num_vertices];
    second_obj_coordinate_y = (float *)&shared_mem[dim * d_max_num_vertices + d_max_num_vertices];
    dot_prod = (float *)&shared_mem[2 * dim * d_max_num_vertices];
    max_index = (unsigned int *)&shared_mem[2 * dim * d_max_num_vertices + d_max_num_vertices];
    direction_vector = (float *)&shared_mem[2 * dim * d_max_num_vertices + 2 * d_max_num_vertices];
    a = (float *)&shared_mem[2 * dim * d_max_num_vertices + 2 * d_max_num_vertices + dim];
    b = (float *)&shared_mem[2 * dim * d_max_num_vertices + 2 * d_max_num_vertices + 2 * dim];
    c = (float *)&shared_mem[2 * dim * d_max_num_vertices + 2 * d_max_num_vertices + 3 * dim];
    d = (float *)&shared_mem[2 * dim * d_max_num_vertices + 2 * d_max_num_vertices + 4 * dim];
    temp_point_1 = (float *)&shared_mem[2 * dim * d_max_num_vertices + 2 * d_max_num_vertices + 5 * dim];
    temp_point_2 = (float *)&shared_mem[2 * dim * d_max_num_vertices + 2 * d_max_num_vertices + 6 * dim];
    simplex_size = (unsigned int *)&shared_mem[2 * dim * d_max_num_vertices + 2 * d_max_num_vertices + 7 * dim];
    finish = (bool *)&shared_mem[2 * dim * d_max_num_vertices + 2 * d_max_num_vertices + 7 * dim + 1];
    result = (bool *)&shared_mem[2 * dim * d_max_num_vertices + 2 * d_max_num_vertices + 7 * dim + 2];
    if (dim == DIM_3) {
        first_obj_coordinate_z = (float *)&shared_mem[2 * d_max_num_vertices];
        second_obj_coordinate_z = (float *)&shared_mem[dim * d_max_num_vertices + 2 * d_max_num_vertices];
    }

    // each block handles one collision
    unsigned long collision_index = blockIdx.x;
    uint32_t first_obj = d_first_obj_index[collision_index];
    uint32_t second_obj = d_second_obj_index[collision_index];
    uint32_t first_obj_num_vertices = d_num_vertices[first_obj];
    uint32_t second_obj_num_vertices = d_num_vertices[second_obj];
    uint32_t first_obj_starting_index = d_obj_starting_index[first_obj];
    uint32_t second_obj_starting_index = d_obj_starting_index[second_obj];

    // bring coordinates data from global mem to shared mem
    bring_data_to_shared_mem(first_obj_coordinate_x, d_vertices_coordinate_x, first_obj_num_vertices, first_obj_starting_index);
    bring_data_to_shared_mem(first_obj_coordinate_y, d_vertices_coordinate_y, first_obj_num_vertices, first_obj_starting_index);
    bring_data_to_shared_mem(second_obj_coordinate_x, d_vertices_coordinate_x, second_obj_num_vertices, second_obj_starting_index);
    bring_data_to_shared_mem(second_obj_coordinate_y, d_vertices_coordinate_y, second_obj_num_vertices, second_obj_starting_index);
    if (dim == DIM_3) {
        bring_data_to_shared_mem(first_obj_coordinate_z, d_vertices_coordinate_z, first_obj_num_vertices, first_obj_starting_index);
        bring_data_to_shared_mem(second_obj_coordinate_z, d_vertices_coordinate_z, second_obj_num_vertices, second_obj_starting_index);
    }
    __syncthreads(); // make sure all data has been brought into shared mem

    // setup the initial values
    unsigned int iteration_counter = 0;
    if (threadIdx.x == 0) {
        direction_vector[0] = 1.0;
        direction_vector[1] = 0.0;
        if (dim == DIM_3) {
            direction_vector[2] = 0.0;
        }
        *simplex_size = 0;
        *finish = false;
        *result = false;
    }
    __syncthreads();
    while (!(*finish) && iteration_counter < 2000) {
        // get the farest point in the Minkowski sum and store it in point a
        support_Minkowski_sum(first_obj_coordinate_x, first_obj_coordinate_y, first_obj_coordinate_z,
                              second_obj_coordinate_x, second_obj_coordinate_y, second_obj_coordinate_z,
                              first_obj_num_vertices, second_obj_num_vertices,
                              direction_vector, dot_prod, max_index, temp_point_1, temp_point_2, a, dim);
        // enter the sequential part, uses only one thread to check if we can encompass the origin or if we have already contains origin
        // and if not, calculate the next direction_vector
        if (threadIdx.x == 0) {
            // check if the farest point in a direction has passed the origin or not
            if (dim == DIM_2 && (a[0] * direction_vector[0] + a[1] * direction_vector[1]) < 0) {
                *result = false;
                *finish = true;
            }
            else if (dim == DIM_3 && (a[0] * direction_vector[0] + a[1] * direction_vector[1] + a[2] * direction_vector[2]) < 0) {
                *result = false;
                *finish = true;
            }
            else {
                // check if we have encompassed the origin, if not update the direction_vector
                update(a, b, c, d, simplex_size, direction_vector, result, finish, dim);
            }
        }
        iteration_counter++;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_collision_results[blockIdx.x] = *result;
    }
}

__device__ void bring_data_to_shared_mem(float * destination, float * source, uint32_t num, uint32_t starting_index) {
    for (uint32_t i = threadIdx.x; i < num; i += blockDim.x) {
        destination[i] = source[starting_index + i];
    }
}

__device__ void support_Minkowski_sum(float * first_obj_coordinate_x,
                                      float * first_obj_coordinate_y,
                                      float * first_obj_coordinate_z,
                                      float * second_obj_coordinate_x,
                                      float * second_obj_coordinate_y,
                                      float * second_obj_coordinate_z,
                                      uint32_t first_obj_num_vertices,
                                      uint32_t second_obj_num_vertices,
                                      float * direction_vector,
                                      float * dot_prod,
                                      unsigned int * max_index,
                                      float * temp_point_1,
                                      float * temp_point_2,
                                      float * point,
                                      unsigned int dim) {

    support_one_object(first_obj_coordinate_x, first_obj_coordinate_y, first_obj_coordinate_z,
                       first_obj_num_vertices, direction_vector, dot_prod, max_index, temp_point_1, dim, false);
    support_one_object(second_obj_coordinate_x, second_obj_coordinate_y, second_obj_coordinate_z,
                       second_obj_num_vertices, direction_vector, dot_prod, max_index, temp_point_2, dim, true);
    if (threadIdx.x == 0) {
        point[0] = temp_point_1[0] - temp_point_2[0];
        point[1] = temp_point_1[1] - temp_point_2[1];
        if (dim == DIM_3) {
            point[2] = temp_point_1[2] - temp_point_2[2];
        }
    }
    __syncthreads();
}

__device__ void support_one_object(float * coordiante_x,
                                   float * coordinate_y,
                                   float * coordinate_z,
                                   uint32_t num_vertices,
                                   float * direction_vector,
                                   float * dot_prod,
                                   unsigned int * max_index,
                                   float * point,
                                   unsigned int dim,
                                   bool negate) {

    if (dim == DIM_2) {
        for (uint32_t i = threadIdx.x; i < num_vertices; i += blockDim.x) {
            dot_prod[i] = coordiante_x[i] * direction_vector[0] + coordinate_y[i] * direction_vector[1];
        }
    }
    else if (dim == DIM_3) {
        for (uint32_t i = threadIdx.x; i < num_vertices; i += blockDim.x) {
            dot_prod[i] = coordiante_x[i] * direction_vector[0] + coordinate_y[i] * direction_vector[1] + coordinate_z[i] * direction_vector[2];
        }
    }
    __syncthreads();
    // for the second object, we need to find the farest point along the opposite direction
    if (negate) {
        for (uint32_t i = threadIdx.x; i < num_vertices; i += blockDim.x) {
            dot_prod[i] = - dot_prod[i];
        }
    }
    __syncthreads();

    // there might be any number of vertices we need to compare and find the index of the max, the steps we are going to take are:
    // 1. find the largest number (2^k) under blockDim.x (including) which is less than num_vertices, we call this number initial_thread_num
    // 2. only uses these initial_thread_num threads to cover num_vertices dot products, for example, if there are 1500 vertices and we use
    //    512 threads, the first thread needs to find the index of the max of the 0, 512, 1024 dot products, the 512th thread needs to find
    //    the index of the max of the 511, 1023 dot products. At the end of this step, we have initial_thread_num indices that we need to
    //    further check.
    // 3. starts with initial_thread_num / 2 threads, each thread compare two dot products and store the index of the bigger on in the threadid
    //    location in max_index array. Then using half of the threads to repeat until we have the index of the largest value.
    // step 1
    unsigned int initial_thread_num = blockDim.x;
    while (initial_thread_num > num_vertices) {
        initial_thread_num /= 2;
    }
    // step 2
    unsigned int tid = threadIdx.x;
    float current = - FLT_MAX;
    float last = - FLT_MAX;
    if (tid < initial_thread_num) {
        for (uint32_t i = tid; i < num_vertices; i += initial_thread_num) {
            current = dot_prod[i];
            if (current > last) {
                max_index[tid] = i;
                last = current;
            }
        }
    }
    __syncthreads();
    // step 3
    for (uint32_t i = initial_thread_num / 2; i > 0; i >>= 1) {
        if (tid < i) {
            if (dot_prod[max_index[tid]] < dot_prod[max_index[tid + i]]) {
                max_index[tid] = max_index[tid + i];
            }
        }
        __syncthreads();
    }
    // at this point, max_index[0] should stores the index of the max dot product
    if (tid == 0) {
        point[0] = coordiante_x[max_index[0]];
        point[1] = coordinate_y[max_index[0]];
        if (dim == DIM_3) {
            point[2] = coordinate_z[max_index[0]];
        }
    }
}

__device__ void update(float * a, float * b, float * c, float * d, unsigned int * simplex_size, float * direction_vector, bool * result, bool * finish,
                       unsigned int dim) {
    // discuss based on the number of points already in the simplex
    if (*simplex_size == 0) {
        // nothing in the simplex, simply add this new point to the simplex and continue
        assign(b, a, dim, false);
        assign(direction_vector, a, dim, true);
        *simplex_size = 1;
        return;
    }
    else if (*simplex_size == 1) {
        float ab[3] = {};
        float ao[3] = {};
        float o[3] = {};
        float cross_ab_ao[3] = {};
        float cross_ab_ao_ab[3] = {};
        vector_subtract(ab, b, a, dim);
        vector_subtract(ao, o, a, dim);
        cross_product(cross_ab_ao, ab, ao);
        cross_product(cross_ab_ao_ab, cross_ab_ao, ab);
        assign(direction_vector, cross_ab_ao_ab, dim, false);
        assign(c, b, dim, false);
        assign(b, a, dim, false);
        *simplex_size = 2;
        return;
    }
    else if (*simplex_size == 2) {
        float o[3] = {};
        float ao[3] = {};
        float ab[3] = {};
        float ac[3] = {};
        float abc[3] = {};
        float abp[3] = {};
        float acp[3] = {};
        vector_subtract(ao, o, a, dim);
        vector_subtract(ab, b, a, dim);
        vector_subtract(ac, c, a, dim);
        cross_product(abc, ab, ac);
        cross_product(abp, ab, abc);
        cross_product(acp, abc, ac);
        if (dot_product(abp, ao) > 0) {
            // the origin lies outside of the triangle, near the edge ab
            // we already have the direction vector that perpendicular to ab, which is abp
            assign(direction_vector, abp, dim, false);
            assign(c, b, dim, false);
            assign(b, a, dim, false);
            return;
        }
        else if (dot_product(acp, ao) > 0) {
            assign(direction_vector, acp, dim, false);
            assign(b, a, dim, false);
            return;
        }
        // if we get to here, the origin must be in the triangle.
        // if dim == 2, then we can stop
        // if dim == 3, we need to check if the origin is below or above the triangle
        if (dim == DIM_2) {
            *result = true;
            *finish = true;
            return;
        }
        else if (dim == DIM_3) {
            // above the triangle
            if (dot_product(abc, ao) > 0) {
                assign(d, c, dim, false);
                assign(c, b, dim, false);
                assign(b, a, dim, false);
                assign(direction_vector, abc, dim, false);
            }
            // below the triangle, we need to reorder the points so that when view from point a to the triangle, points b, c, d are in counter-clockwise order
            else {
                assign(d, b, dim, false);
                assign(b, a, dim, false);
                assign(direction_vector, abc, dim, true); // if not in the abc direction, should be in the opposite direction
            }
            *simplex_size = 3;
            return;
        }
    }
    else if (*simplex_size == 3) {
        float o[3] = {};
        float ao[3] = {};
        float ab[3] = {};
        float ac[3] = {};
        float ad[3] = {};
        float abc[3] = {};
        float acd[3] = {};
        float adb[3] = {};

        vector_subtract(ao, o, a, dim);
        vector_subtract(ab, b, a, dim);
        vector_subtract(ac, c, a, dim);
        vector_subtract(ad, d, a, dim);
        cross_product(abc, ab, ac);
        cross_product(acd, ac, ad);
        cross_product(adb, ad, ab);

        unsigned int plane_test = (dot_product(abc, ao) > 0 ? OVER_ABC : 0) |
                                  (dot_product(acd, ao) > 0 ? OVER_ACD : 0) |
                                  (dot_product(adb, ao) > 0 ? OVER_ADB : 0);

        switch (plane_test) {
            case 0:
                *finish = true;
                *result = true;
                break;
            case OVER_ABC:
                check_one_face(ab, ac, ao, abc, a, b, c, d, direction_vector, simplex_size, dim);
                break;
            case OVER_ACD:
                check_one_face(ac, ad, ao, acd, a, c, d, b, direction_vector, simplex_size, dim);
                break;
            case OVER_ADB:
                check_one_face(ad, ab, ao, adb, a, d, b, c, direction_vector, simplex_size, dim);
                break;
            case OVER_ABC | OVER_ACD:
                check_two_face(ab, ac, ad, ao, abc, acd, a, b, c, d, direction_vector, simplex_size, dim);
                break;
            case OVER_ACD | OVER_ADB:
                check_two_face(ac, ad, ab, ao, acd, adb, a, c, d, b, direction_vector, simplex_size, dim);
                break;
            case OVER_ADB | OVER_ABC:
                check_two_face(ad, ab, ac, ao, adb, abc, a, d, b, c, direction_vector, simplex_size, dim);
                break;
            default:
                *finish = true;
                *result = true;
                break;
        }
    }
}

__device__ void assign(float * left, float * right, unsigned int dim, bool negate) {
    if (!negate) {
        left[0] = right[0];
        left[1] = right[1];
        if (dim == DIM_3) {
            left[2] = right[2];
        }
    }
    else {
        left[0] = -right[0];
        left[1] = -right[1];
        if (dim == DIM_3) {
            left[2] = -right[2];
        }
    }
}

__device__ void vector_subtract(float * result, float * head, float * tail, unsigned int dim) {
    result[0] = head[0] - tail[0];
    result[1] = head[1] - tail[1];
    if (dim == DIM_3) {
        result[2] = head[2] - tail[2];
    }
}

// all cross product are in 3D, we need to prepare the vectors before the cross product
__device__ void cross_product(float * result, float * a, float * b) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

// all dot product are in 3D, we need to prepare the vectors before the dot product
__device__ float dot_product(float * a, float * b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ void check_one_face(float * ab, float * ac, float * ao, float * abc,
                               float * local_a, float * local_b, float * local_c, float * local_d,
                               float * direction_vector, unsigned int * simplex_size, unsigned int dim) {

    float temp_1[3] = {};
    float temp_2[3] = {};
    // check if in the region of ac
    cross_product(temp_1, abc, ac);
    cross_product(temp_2, ab, abc);
    if (dot_product(temp_1, ao) > 0) {
        // in the region of ac
        cross_product(temp_1, ac, ao);
        cross_product(temp_2, temp_1, ac);
        assign(direction_vector, temp_2, dim, false);
        assign(local_b, local_a, dim, false);
        *simplex_size = 2;
        return;
    }
    else if (dot_product(temp_2, ao) > 0) {
        // in the region of ab
        cross_product(temp_1, ab, ao);
        cross_product(temp_2, temp_1, ab);
        assign(direction_vector, temp_2, dim, false);
        assign(local_c, local_b, dim, false);
        assign(local_b, local_a, dim, false);
        *simplex_size = 2;
        return;
    }
    else {
        // in the region of abc
        assign(direction_vector, abc, dim, false);
        assign(local_d, local_c, dim, false);
        assign(local_c, local_b, dim, false);
        assign(local_b, local_a, dim, false);
        *simplex_size = 3;
        return;
    }
}

__device__ void check_two_face(float * ab, float * ac, float * ad, float * ao, float * abc, float * acd,
                               float * local_a, float * local_b, float * local_c, float * local_d,
                               float * direction_vector, unsigned int * simplex_size, unsigned int dim) {

    float temp_1[3] = {};
    float temp_2[3] = {};
    // check if in the region of ac
    cross_product(temp_1, abc, ac);
    cross_product(temp_2, ab, abc);
    if (dot_product(temp_1, ao) > 0) {
        // in the region of ac, we need to continue check what's the position of origin relative to plane acd
        check_one_face(ac, ad, ao, acd, local_a, local_c, local_d, local_b, direction_vector, simplex_size, dim);
        return;
    }
    else if (dot_product(temp_2, ao) > 0) {
        // in the region of ab
        cross_product(temp_1, ab, ao);
        cross_product(temp_2, temp_1, ab);
        assign(direction_vector, temp_2, dim, false);
        assign(local_c, local_b, dim, false);
        assign(local_b, local_a, dim, false);
        *simplex_size = 2;
        return;
    }
    else {
        // in the region of abc
        assign(local_d, local_c, dim, false);
        assign(local_c, local_b, dim, false);
        assign(local_b, local_a, dim, false);
        assign(direction_vector, abc, dim, false);
        *simplex_size = 3;
        return;
    }
}
