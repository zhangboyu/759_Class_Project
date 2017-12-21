#ifndef COLLISION_CPU_NARROW
#define COLLISION_CPU_NARROW 1

#include <cfloat>
#include <cstdlib>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "globalVariables.h"

void test_collision_gpu(unsigned long num_objects,
                        unsigned long num_collision,
                        uint32_t * d_first_obj_index,
                        uint32_t * d_second_obj_index,
                        uint32_t * d_num_vertices,
                        float * d_vertices_coordinate_x,
                        float * d_vertices_coordinate_y,
                        float * d_vertices_coordinate_z,
                        unsigned int dim,
                        bool * d_collision_results,
                        float * time);

#endif
