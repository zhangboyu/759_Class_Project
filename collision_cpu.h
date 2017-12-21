#ifndef COLLISION_CPU
#define COLLISION_CPU 1

#include <cstdlib>
#include <iostream>
#include <cuda.h>

unsigned int count_sollision_cpu(float * obj_coordinates, float * obj_radii, unsigned int dim, unsigned long num_objects, float * time);

#endif
