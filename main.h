#ifndef MAIN_H
#define MAIN_H 1

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <vector>
#include <sstream>
#include <random>

#include "globalVariables.h"
#include "collision_cpu.h"
#include "collision_gpu_broad.h"
#include "collision_gpu_narrow.h"

float obj_init(float * coordinates, float * radii, unsigned long num_objects, unsigned int dim, int test_flag);
float bounding_sphere_init(float * obj_radii, float * bounding_sphere_radii, unsigned long num_objects);
void print_obj_info(float * obj_coordinates, float * obj_radii, unsigned int dim, unsigned long num_objects);
void init_obj_index(unsigned int * obj_indices, unsigned int * obj_num_vertices, unsigned long num_objects, unsigned int available_obj_model);
template<typename Out>
void split(const std::string &s, char delim, Out result);
std::vector<std::string> split(const std::string &s, char delim);
void updateBoundary(float value, float *max, float *min);
void shiftObject(double shiftDistance,double shiftDistance2,double shiftDistance3, int startIndex, int endIndex, float* x, float* y, float* z);
double randomNumber (double upperLimit);
void readFile(float* xCoord, float* yCoord, float* zCoord, const std::string fileName, int dimension, int startIndex,
              float& xCentroid, float& yCentroid, float& zCentroid, float& objectLength);
void preProcess(float* x, float* y, float* z, float* Centroid,
                float* obj_radii, unsigned int* obj_indices, unsigned int num_objects, double space_scaling);

#endif
