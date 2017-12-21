#include "collision_cpu.h"

unsigned int count_sollision_cpu(float * obj_coordinates, float * obj_radii, unsigned int dim, unsigned long num_objects, float * time) {

    cudaEvent_t startEvent_inc, stopEvent_inc;
    cudaEventCreate(&startEvent_inc);
    cudaEventCreate(&stopEvent_inc);
    cudaEventRecord(startEvent_inc,0);

    unsigned long collisions = 0;
    float current_radius, radius_sum, coordinate_diff, distance;

    for (size_t j = 0; j < num_objects; j++) {
        current_radius =  obj_radii[j];

        for (size_t k = j + 1; k < num_objects; k++) {

            // assume  obj_radii are radii of balls
            radius_sum =  obj_radii[k] + current_radius;
            distance = 0;

            for (size_t l = 0; l < dim; l++) {
                coordinate_diff =  obj_coordinates[j + l * num_objects] - obj_coordinates[k + l * num_objects];
                distance += coordinate_diff * coordinate_diff;
            }
            // std::cout << "Comparing between " << j << " and " << k << '\n';
            // std::cout << "distance = " << distance << '\n';
            // if collision
            if (distance < radius_sum * radius_sum) {
                collisions++;
            }
        }
    }

    cudaEventRecord(stopEvent_inc,0);
    cudaEventSynchronize(stopEvent_inc);
    cudaEventElapsedTime(time, startEvent_inc, stopEvent_inc);
    return collisions;
}
