#ifndef COLLISION_GPU
#define COLLISION_GPU 1

#include <cstdlib>
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/scan.h>
#include <unordered_set>

#include "globalVariables.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct collision_pair_compare {
    bool operator()(const std::pair<uint32_t, uint32_t> & p1, const std::pair<uint32_t, uint32_t> & p2) const {
        return (p1.first == p2.first && p1.second == p2.second) || (p1.first == p2.second && p1.second == p2.first);
    }
};

struct pair_hashing {
    std::size_t operator()(const std::pair<uint32_t, uint32_t> & p) const {
        std::size_t h1 = std::hash<uint32_t>{}(p.first);
        std::size_t h2 = std::hash<uint32_t>{}(p.second);
        return h1 ^ h2;
    }
};

unsigned long count_collision_gpu(float * obj_coordinates,
                                  float * bs_radii,
                                  float max_bs_radius,
                                  unsigned int dim,
                                  unsigned long num_objects,
                                  uint32_t ** first_obj_index,
                                  uint32_t ** second_obj_index,
                                  float * total_time);
unsigned long cell_array_init(float * obj_coordinates,
                              float * bs_radii,
                              float max_bs_radius,
                              unsigned long num_objects,
                              unsigned int dim,
                              unsigned int max_touch_cell_num,
                              unsigned int neighbor_cell_num,
                              uint32_t * d_cell_array,
                              uint32_t * d_obj_array,
                              uint32_t * d_valid_cell_counters,
                              float * time);
unsigned long perform_collision_test(uint32_t * d_cell_array,
                                     uint32_t * d_obj_array,
                                     unsigned long valid_cell_num,
                                     unsigned long num_objects,
                                     unsigned int max_touch_cell_num,
                                     uint32_t ** first_obj_index,
                                     uint32_t ** second_obj_index,
                                     float * time);

#endif
