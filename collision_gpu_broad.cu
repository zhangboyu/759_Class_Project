#include "collision_gpu_broad.h"

__global__ void cuda_cell_array_init(float * d_obj_coordinates,
                                     float * d_bs_radii,
                                     float cell_size,
                                     unsigned long num_objects,
                                     unsigned int dim,
                                     unsigned int max_touch_cell_num,
                                     unsigned int neighbor_cell_num,
                                     uint32_t * d_cell_array,
                                     uint32_t * d_obj_array,
                                     uint32_t * d_valid_cell_counters);

__global__ void cuda_collision_test(uint32_t * d_cell_array,
                                    uint32_t * d_obj_array,
                                    unsigned long valid_cell_num,
                                    unsigned long cells_per_thread,
                                    uint32_t * d_starting_indices,
                                    uint32_t * d_tailing_cell_num,
                                    unsigned long * d_collision_per_thread);

__global__ void cuda_get_obj_indices(uint32_t * d_obj_array,
                                     unsigned long * d_collision_per_thread,
                                     unsigned long * d_collision_starting_index,
                                     unsigned long cells_per_thread,
                                     uint32_t * d_starting_indices,
                                     uint32_t * d_tailing_cell_num,
                                     uint32_t * d_first_obj_index,
                                     uint32_t * d_second_obj_index);

unsigned long count_collision_gpu(float * obj_coordinates,
                                  float * bs_radii,
                                  float max_bs_radius,
                                  unsigned int dim,
                                  unsigned long num_objects,
                                  uint32_t ** d_first_obj_index,
                                  uint32_t ** d_second_obj_index,
                                  float * total_time) {

    unsigned long num_collision = 0;
    unsigned int max_touch_cell_num = 0;
    unsigned int neighbor_cell_num = 0;

    float time_part1, time_part2, time_part3;

    if (dim == DIM_2) {
        max_touch_cell_num = MAX_TOUCH_CELL_DIM_2;
        neighbor_cell_num = NEIGHBOR_CELL_DIM_2;
    }
    else if(dim == DIM_3) {
        max_touch_cell_num = MAX_TOUCH_CELL_DIM_3;
        neighbor_cell_num = NEIGHBOR_CELL_DIM_3;
    }

    unsigned long cell_array_size = max_touch_cell_num * num_objects;
    unsigned long obj_array_size = max_touch_cell_num * num_objects;
    unsigned long obj_coor_size = dim * num_objects;

    // d_cell_array holds the cell array info for each object
    // d_obj_array holds the object info for each entry in the cell array
    float * d_obj_coordinates, * d_bs_radii;
    uint32_t * d_cell_array, * d_obj_array, * d_valid_cell_counters;// * d_first_obj_index, * d_second_obj_index;
    // uint32_t * first_obj_index, * second_obj_index;
    cudaMalloc((void **)&d_cell_array, cell_array_size * sizeof(uint32_t));
    cudaMalloc((void **)&d_obj_array, obj_array_size * sizeof(uint32_t));
    cudaMalloc((void **)&d_valid_cell_counters, num_objects * sizeof(uint32_t));
    cudaMalloc((void **)&d_obj_coordinates, obj_coor_size * sizeof(float));
    cudaMalloc((void **)&d_bs_radii, num_objects * sizeof(float));
    cudaMemcpy(d_obj_coordinates, obj_coordinates, obj_coor_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bs_radii, bs_radii, num_objects * sizeof(float), cudaMemcpyHostToDevice);

    // thrust::device_ptr<uint32_t> cell_array = thrust::device_pointer_cast(d_cell_array);
    // thrust::device_ptr<uint32_t> obj_array = thrust::device_pointer_cast(d_obj_array);
    // thrust::device_ptr<uint32_t> valid_cell_counters = thrust::device_pointer_cast(d_valid_cell_counters);

    unsigned long valid_cell_num = cell_array_init(d_obj_coordinates,
                                                   d_bs_radii,
                                                   max_bs_radius,
                                                   num_objects,
                                                   dim,
                                                   max_touch_cell_num,
                                                   neighbor_cell_num,
                                                   d_cell_array,
                                                   d_obj_array,
                                                   d_valid_cell_counters,
                                                   &time_part1);


    cudaEvent_t startEvent_inc, stopEvent_inc;
    cudaEventCreate(&startEvent_inc);
    cudaEventCreate(&stopEvent_inc);
    cudaEventRecord(startEvent_inc,0);

    thrust::stable_sort_by_key(thrust::device, d_cell_array, d_cell_array + cell_array_size, d_obj_array);

    cudaEventRecord(stopEvent_inc,0);
    cudaEventSynchronize(stopEvent_inc);
    cudaEventElapsedTime(&time_part2, startEvent_inc, stopEvent_inc);


    num_collision = perform_collision_test(d_cell_array,
                                           d_obj_array,
                                           valid_cell_num,
                                           num_objects,
                                           max_touch_cell_num,
                                           d_first_obj_index,
                                           d_second_obj_index,
                                           &time_part3);

    *total_time = time_part1 + time_part2 + time_part3;
    // here, there might be some duplications in obj_index array, for example, obj pair 1 & 2 might appear like
    // 1 & 2 and 2 & 1
    // first_obj_index = (uint32_t *)malloc(num_collision * sizeof(uint32_t));
    // second_obj_index = (uint32_t *)malloc(num_collision * sizeof(uint32_t));
    // cudaMemcpy(first_obj_index, d_first_obj_index, num_collision * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(second_obj_index, d_second_obj_index, num_collision * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // std::unordered_set<std::pair<uint32_t, uint32_t>, pair_hashing, collision_pair_compare> collision_pairs;
    // for (size_t i = 0; i < num_collision; i++) {
    //     auto result = collision_pairs.insert(std::make_pair(first_obj_index[i], second_obj_index[i]));
    // }
    // std::cout << "Number of collisions detected by GPU (Before de-duplicate) = " << num_collision << '\n';
    // std::cout << "Number of collisions detected by GPU = " << collision_pairs.size() << '\n';
    // std::cout << "Valid cell num = " << valid_cell_num << '\n';

    cudaFree(d_cell_array);
    cudaFree(d_obj_array);
    cudaFree(d_valid_cell_counters);
    cudaFree(d_obj_coordinates);
    cudaFree(d_bs_radii);
    return num_collision;
}

unsigned long cell_array_init(float * d_obj_coordinates,
                              float * d_bs_radii,
                              float max_bs_radius,
                              unsigned long num_objects,
                              unsigned int dim,
                              unsigned int max_touch_cell_num,
                              unsigned int neighbor_cell_num,
                              uint32_t * d_cell_array,
                              uint32_t * d_obj_array,
                              uint32_t * d_valid_cell_counters,
                              float * time) {

    float cell_size = CELL_SIZE_SCALING * max_bs_radius * 2; // *2 turn radius to diameter
    unsigned int thread_per_block = 512; // this value is set to 512 because we will run out of registers if set to 1024
    unsigned long num_block = (num_objects + thread_per_block - 1) / thread_per_block;
    cudaMemset(d_cell_array, INVALID_CELL, max_touch_cell_num * num_objects * sizeof(uint32_t));
    cudaMemset(d_obj_array, INVALID_CELL, max_touch_cell_num * num_objects * sizeof(uint32_t));
    cudaMemset(d_valid_cell_counters, 0, num_objects * sizeof(uint32_t));

    cudaEvent_t startEvent_inc, stopEvent_inc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
	cudaEventRecord(startEvent_inc,0);
    cuda_cell_array_init<<<num_block, thread_per_block>>>(d_obj_coordinates,
                                                          d_bs_radii,
                                                          cell_size,
                                                          num_objects,
                                                          dim,
                                                          max_touch_cell_num,
                                                          neighbor_cell_num,
                                                          d_cell_array,
                                                          d_obj_array,
                                                          d_valid_cell_counters);
    cudaEventRecord(stopEvent_inc,0);
  	cudaEventSynchronize(stopEvent_inc);
  	cudaEventElapsedTime(time, startEvent_inc, stopEvent_inc);

    return thrust::reduce(thrust::device, d_valid_cell_counters, d_valid_cell_counters + num_objects);
}

unsigned long perform_collision_test(uint32_t * d_cell_array,
                                     uint32_t * d_obj_array,
                                     unsigned long valid_cell_num,
                                     unsigned long num_objects,
                                     unsigned int max_touch_cell_num,
                                     uint32_t ** first_obj_index,
                                     uint32_t ** second_obj_index,
                                     float * time) {

    // since the entire space is divided into 256 * 256 * 256 sub-space, if the objects are distributed
    // evenly in the whole space, then the number of objects associated with a cell is:
    // num_objects / (256 * 256 * 256) * max_touch_cell_num (3D).
    // num_objects / (256 * 256) * max_touch_cell_num (2D).
    // * max_touch_cell_num is because an object can be associated with at most max_touch_cell_num cells.
    // so we set the average number of cells processed by each thread to be:
    // num_objects / (256 * 256 * 256 / max_touch_cell_num) + 1 (3D)
    // num_objects / (256 * 256 / max_touch_cell_num) + 1 (2D)
    // note that the choice of this number shouldn't affect the correctness of the program but it will
    // affect how many blocks / threads are launched and finally runtime
    unsigned long cells_per_thread = 0;
    if (max_touch_cell_num == MAX_TOUCH_CELL_DIM_2) {
        cells_per_thread = num_objects / (MAX_CELL_NUM_ANY_DIM * MAX_CELL_NUM_ANY_DIM / max_touch_cell_num) + 1;
    }
    else if (max_touch_cell_num == MAX_TOUCH_CELL_DIM_3) {
        cells_per_thread = num_objects / (MAX_CELL_NUM_ANY_DIM * MAX_CELL_NUM_ANY_DIM * MAX_CELL_NUM_ANY_DIM / max_touch_cell_num) + 1;
    }
    // unsigned long cells_per_thread = 60;
    // like above, we set the number of thread per block to 512
    unsigned long thread_per_block = 512;
    unsigned long num_threads_needed = (valid_cell_num + cells_per_thread - 1) / cells_per_thread;
    unsigned long num_block = (num_threads_needed + thread_per_block - 1) / thread_per_block;
    unsigned long * d_collision_per_thread, * d_collision_starting_index; // this array holds the position in the first/second_obj_index array
                                                                          // where each thread should write the collision it detected
    uint32_t * d_starting_indices, * d_tailing_cell_num; // these two arrays hold the info about the index of the home cell and how many cells after it
                                                         // we consider a collision
    cudaMalloc((void **)&d_collision_per_thread, num_block * thread_per_block * sizeof(unsigned long));
    cudaMalloc((void **)&d_collision_starting_index, num_block * thread_per_block * sizeof(unsigned long));
    cudaMalloc((void **)&d_starting_indices, num_block * thread_per_block * cells_per_thread * sizeof(uint32_t));
    cudaMalloc((void **)&d_tailing_cell_num, num_block * thread_per_block * cells_per_thread * sizeof(uint32_t));
    cudaMemset(d_collision_per_thread, 0, num_block * thread_per_block * sizeof(unsigned long));
    cudaMemset(d_collision_starting_index, 0, num_block * thread_per_block * sizeof(unsigned long));
    cudaMemset(d_starting_indices, INVALID_CELL, num_block * thread_per_block * cells_per_thread * sizeof(uint32_t));
    cudaMemset(d_tailing_cell_num, 0, num_block * thread_per_block * cells_per_thread * sizeof(uint32_t));

    cudaEvent_t startEvent_inc, stopEvent_inc;
    cudaEventCreate(&startEvent_inc);
    cudaEventCreate(&stopEvent_inc);
    cudaEventRecord(startEvent_inc,0);

    cuda_collision_test<<<num_block, thread_per_block>>>(d_cell_array,
                                                         d_obj_array,
                                                         valid_cell_num,
                                                         cells_per_thread,
                                                         d_starting_indices,
                                                         d_tailing_cell_num,
                                                         d_collision_per_thread);
    // the number of collision detected by each thread is stored in array d_collision_per_thread
    // so the total number of collisions is a reduce of this array with addition
    // for each collision, we need to find the objects index and store somewhere, and we are going to use the same number of thread to
    // perform this task, so there is a one to one mapping, the collisions found by a thead will be handled by a thread.
    // We use a exclusive scan to find the index of the obj info of the first collision of a thread.
    unsigned long num_collision = thrust::reduce(thrust::device, d_collision_per_thread, d_collision_per_thread + num_block * thread_per_block);
    thrust::exclusive_scan(thrust::device, d_collision_per_thread, d_collision_per_thread + num_block * thread_per_block, d_collision_starting_index);

    uint32_t * d_first_obj_index, * d_second_obj_index;
    cudaMalloc((void **)&d_first_obj_index, num_collision * sizeof(uint32_t));
    cudaMalloc((void **)&d_second_obj_index, num_collision * sizeof(uint32_t));
    cuda_get_obj_indices<<<num_block, thread_per_block>>>(d_obj_array,
                                                          d_collision_per_thread,
                                                          d_collision_starting_index,
                                                          cells_per_thread,
                                                          d_starting_indices,
                                                          d_tailing_cell_num,
                                                          d_first_obj_index,
                                                          d_second_obj_index);

    cudaEventRecord(stopEvent_inc,0);
    cudaEventSynchronize(stopEvent_inc);
    cudaEventElapsedTime(time, startEvent_inc, stopEvent_inc);

    * first_obj_index = d_first_obj_index;
    * second_obj_index = d_second_obj_index;
    // thrust::device_ptr<unsigned long> collision_per_thread = thrust::device_pointer_cast(d_collision_per_thread);
    cudaFree(d_collision_per_thread);
    cudaFree(d_collision_starting_index);
    cudaFree(d_starting_indices);
    cudaFree(d_tailing_cell_num);
    return num_collision;
}

__global__ void cuda_cell_array_init(float * d_obj_coordinates,
                                     float * d_bs_radii,
                                     float cell_size,
                                     unsigned long num_objects,
                                     unsigned int dim,
                                     unsigned int max_touch_cell_num,
                                     unsigned int neighbor_cell_num,
                                     uint32_t * d_cell_array,
                                     uint32_t * d_obj_array,
                                     uint32_t * d_valid_cell_counters) {

    unsigned long threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < num_objects) {
        uint32_t hash = 0;
        uint32_t sides = 0;
        unsigned int valid_cell_counter = 0;
        unsigned int neighbor_cell_index = 0;
        int neighbor_cell_offset = 0;
        float bs_radius = d_bs_radii[threadId];
        float obj_coordinate, obj_coordinate_in_cell;

        // find home cell
        for (size_t j = 0; j < dim; j++) {
            obj_coordinate = d_obj_coordinates[threadId + j * num_objects];   // this is the coordiante of the object in the jth dimension

            // cell ID is simply the bits of each cell coordinate concatenated
            // only shift 8 bits, which means the whole space only be divided into at most 256 sub-space in each dimension
            // which again means the longest dimension of the largest object cannot smaller than 1/256 of the whole space
            hash = hash << 8 | (uint32_t) (obj_coordinate / cell_size);

            // determine if the object is close enogh to touch cell walls
            obj_coordinate_in_cell = obj_coordinate - floor(obj_coordinate / cell_size) * cell_size;

            // keep track of which side of the cell, if any, the object overlaps
            if (obj_coordinate_in_cell < bs_radius) {  // object touches the floor of home cell
                sides = (sides << 2) | 0x03;
            }
            else if (cell_size - obj_coordinate_in_cell < bs_radius) {  // object touches the ceiling of home cell
                sides = (sides << 2) | 0x01;
            }
            else {
                sides <<= 2;
            }
        }

        // bit 0 unset indicates home cell
        d_cell_array[threadId] = hash;
        d_obj_array[threadId] = threadId << 1 | 0x01;
        valid_cell_counter++;

        // find phantom cells in the Moore neighborhood
        // here neighbor_cell_num = 9 represents the 8 immediate neighbors around a cell and the cell itself (in 2D layout)
        for (size_t j = 0; j < neighbor_cell_num; j++) {
            // skip the home (center) cell since it's already been added
            // in 2D layout the home cell is at index 4
            // -------
            // |6|7|8|
            // -------
            // |3|4|5|
            // -------
            // |0|1|2|
            // -------
            if (j == neighbor_cell_num / 2) {
                continue;
            }

            // run through the components of each potential side cell
            neighbor_cell_index = j;
            hash = 0;

            for (size_t k = 0; k < dim; k++) {
                // r represents the offset of current cell relative to the home cell in current (kth) dimension, notice at the end of this loop, q is divided by 3
                neighbor_cell_offset = neighbor_cell_index % 3 - 1;
                // coordinate of current object in current dimension
                obj_coordinate = d_obj_coordinates[threadId + k * num_objects];

                // skip this cell if the object is on the wrong side
                // this if statement checks if we are sure that the object will not touch current cell
                // when r == 0, it means current cell is parallel with the home cell in current dimension, like 1 and 7 to 4 in horizontal, 3 and 5 to 4 in vertical
                // when r == -1, it represents the cells at -1 offset to home cell in current dimension, like 0 3 6 to 4 in horizontal and 0 1 2 to 4 in vertical
                // when r == 1,  it represents the cells at 1 offset to home cell in current dimension, like 2 5 8 to 4 in horizontal and 6 7 8 to 4 in vertical
                // so when r == 0, the check is false means we don't know if the object will overlap with current cell or not and we need to check other dimensions
                // when r == -1, it checks the floor info remembered in the previous step (floor 0x03)
                // when r == 1, it checks the ceiling info remembered in the previous step (ceiling 0x01)
                // for example, in cell 6, we need to check if the object touches cell 4's floor and cell 4's ceiling, if it touches both, it is highly likely it
                // touches cell 6, however, it is also possible that the object teouch 3 and 7 but not 6, the code just doesn't handle that case, in that case
                // we will just handle more potential collide pairs to narrow-phase but the correctness is not affected.
                // (sides >> ((dim - 1 - k) * 2) shifts "sides" to the point that the lower 2 bits are the info we care about at this dimention
                // & 0x03 masks the other bits out so that we only left with the two bits we care about
                // ^ r compares between the sides info we get to the current cell's position we care about
                // & 0x03 again masks out any other bits
                if ((neighbor_cell_offset != 0 && (sides >> ((dim - 1 - k) * 2) & 0x03 ^ neighbor_cell_offset) & 0x03) ||
                    (((d_cell_array[threadId] >> ((dim - 1 - k) * 8)) & 0x000000FF) == 0x00000000 && neighbor_cell_offset == -1) ||
                    (((d_cell_array[threadId] >> ((dim - 1 - k) * 8)) & 0x000000FF) == 0x000000FF && neighbor_cell_offset == 1)) {
                    // these two conditions test if the object is at the the wall of whole space so the cells that are out of the whole space are ignored
                    hash = INVALID_CELL;
                    break;
                }

                // cell ID of the neighboring cell
                hash = hash << 8 | ((uint32_t) (obj_coordinate / cell_size) + neighbor_cell_offset);
                neighbor_cell_index /= 3;
            }

            // only add this cell to the list if there's potential overlap
            if (hash != INVALID_CELL) {
                d_cell_array[threadId + valid_cell_counter * num_objects] = hash;
                // bit 0 set indicates phantom cell
                d_obj_array[threadId + valid_cell_counter * num_objects] = threadId << 1 | 0x00;

                // count the number of valid cells in the cell array for current object
                valid_cell_counter++;
            }
        }
        d_valid_cell_counters[threadId] = valid_cell_counter;
    }
}

__global__ void cuda_collision_test(uint32_t * d_cell_array,
                                    uint32_t * d_obj_array,
                                    unsigned long valid_cell_num,
                                    unsigned long cells_per_thread,
                                    uint32_t * d_starting_indices,
                                    uint32_t * d_tailing_cell_num,
                                    unsigned long * d_collision_per_thread) {

    unsigned long threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long thread_start_index = threadId * cells_per_thread; // inclusive
    unsigned long thread_end_index = thread_start_index + cells_per_thread > valid_cell_num ? valid_cell_num : thread_start_index + cells_per_thread;  // exclusive
    if (thread_start_index < valid_cell_num) {
        uint32_t last = INVALID_CELL;
        unsigned long home_cell_num = 0;
        unsigned long phantom_cell_num = 0;
        unsigned long current_index = thread_start_index;
        unsigned long starting_home_index = 0;
        unsigned int  starting_indices_counter = 0;

        while ((current_index < thread_end_index || d_cell_array[current_index] == last) && current_index < valid_cell_num) {
            if (d_cell_array[current_index] != last && d_obj_array[current_index] & 0x01) {
                starting_home_index = current_index;
                home_cell_num++;
            }
            else if (d_cell_array[current_index] == last && home_cell_num != 0) {
                if (d_obj_array[current_index] & 0x01 && current_index < thread_end_index) {
                    home_cell_num++; // here the home_cell_num is only the number of home cells within the region covered by current thread
                }
                else {
                    phantom_cell_num++; // here the phantom_cell_num contains not only phantom cells but also the home cells not covered
                                        // by current thread, which are deemed as phantom cells by current thread
                }
            }
            last = d_cell_array[current_index];
            current_index++;

            if (d_cell_array[current_index] != last) {
                if (home_cell_num != 0 && home_cell_num + phantom_cell_num > 1) {
                    for (size_t i = 0; i < home_cell_num; i++) {
                        d_starting_indices[threadId * cells_per_thread + starting_indices_counter] = starting_home_index + i;
                        d_tailing_cell_num[threadId * cells_per_thread + starting_indices_counter] = phantom_cell_num + home_cell_num - 1 - i;
                        d_collision_per_thread[threadId] += phantom_cell_num + home_cell_num - 1 - i;
                        starting_indices_counter++;
                    }
                }
                home_cell_num = 0;
                phantom_cell_num = 0;
            }
        }
    }
}

__global__ void cuda_get_obj_indices(uint32_t * d_obj_array,
                                     unsigned long * d_collision_per_thread,
                                     unsigned long * d_collision_starting_index,
                                     unsigned long cells_per_thread,
                                     uint32_t * d_starting_indices,
                                     uint32_t * d_tailing_cell_num,
                                     uint32_t * d_first_obj_index,
                                     uint32_t * d_second_obj_index) {

    unsigned long threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_collision_per_thread[threadId] > 0) {
        unsigned long position = d_collision_starting_index[threadId];
        for (size_t i = 0; i < cells_per_thread; i++) {
            unsigned long starting_cell_index = d_starting_indices[threadId * cells_per_thread + i];
            if (starting_cell_index != INVALID_CELL) {
                for (size_t j = 1; j <= d_tailing_cell_num[threadId * cells_per_thread + i]; j++) {
                    d_first_obj_index[position] = d_obj_array[starting_cell_index] >> 1;
                    d_second_obj_index[position] = d_obj_array[starting_cell_index + j] >> 1;
                    position++;
                }
            }
        }
    }
}
