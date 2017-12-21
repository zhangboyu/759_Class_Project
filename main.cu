#include "main.h"

int main(int argc, char const *argv[]) {
    using namespace std;
    // parse the input arguments
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " DIM(2/3) NUM_OBJECTS SPACE_SCALING" << '\n';
        return -1;
    }
    unsigned int current_dim = atoi(argv[1]);
    unsigned long num_objects = atoi(argv[2]);
    double space_scaling = atof(argv[3]);
    if (space_scaling > 1) {
        space_scaling = 1;
    }
    // unsigned int  test_flag = 0;
    // if (argc >= 5) {
    //     test_flag = atoi(argv[4]);
    // }

    // setup the memorys
    uint32_t available_obj_model = 1;
    uint32_t * obj_indices = (unsigned int *)malloc(num_objects * sizeof(unsigned int));
    uint32_t * obj_num_vertices = (unsigned int *)malloc(num_objects * sizeof(unsigned int));
    init_obj_index(obj_indices, obj_num_vertices, num_objects, available_obj_model);
    unsigned long obj_total_vertices = std::accumulate(obj_num_vertices, obj_num_vertices + num_objects, 0);
    float * vertices_coordinate_x = (float *)malloc(obj_total_vertices * sizeof(float));
    float * vertices_coordinate_y = (float *)malloc(obj_total_vertices * sizeof(float));
    float * vertices_coordinate_z = (float *)malloc(obj_total_vertices * sizeof(float));
    float * obj_center_coordinates = (float *)malloc(current_dim * num_objects * sizeof(float));
    float * obj_radii = (float *)malloc(num_objects * sizeof(float));
    float * bs_radii = (float *)malloc(num_objects * sizeof(float));

    // fill up memory
    preProcess(vertices_coordinate_x, vertices_coordinate_y, vertices_coordinate_z, obj_center_coordinates, obj_radii, obj_indices, num_objects, space_scaling);
    // float max_obj_radius = obj_init(obj_center_coordinates, obj_radii, num_objects, current_dim, test_flag);
    float max_bs_radius = bounding_sphere_init(obj_radii, bs_radii, num_objects);

    uint32_t * d_first_obj_index, * d_second_obj_index; // the space is allocated in function count_collision_gpu
                                                        // we need to free it at here
    float * d_vertices_coordinate_x, * d_vertices_coordinate_y, * d_vertices_coordinate_z;
    bool * d_collision_results, * collision_results;
    uint32_t * d_num_vertices;
    float gpu_time_broad, gpu_time_narrow, cpu_time_broad;

    unsigned long num_collisions_cpu_broad = count_sollision_cpu(obj_center_coordinates, obj_radii, current_dim, num_objects, &cpu_time_broad);
    unsigned long num_collisions_gpu_broad = count_collision_gpu(obj_center_coordinates, bs_radii, max_bs_radius, current_dim,
                                                           num_objects, &d_first_obj_index, &d_second_obj_index, &gpu_time_broad);

    cudaMalloc((void **)&d_num_vertices, num_objects * sizeof(uint32_t));
    cudaMalloc((void **)&d_vertices_coordinate_x, obj_total_vertices * sizeof(float));
    cudaMalloc((void **)&d_vertices_coordinate_y, obj_total_vertices * sizeof(float));
    cudaMalloc((void **)&d_vertices_coordinate_z, obj_total_vertices * sizeof(float));
    cudaMalloc((void **)&d_collision_results, num_collisions_gpu_broad * sizeof(bool));
    cudaMemcpy(d_num_vertices, obj_num_vertices, num_objects * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertices_coordinate_x, vertices_coordinate_x, obj_total_vertices * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertices_coordinate_y, vertices_coordinate_y, obj_total_vertices * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertices_coordinate_z, vertices_coordinate_z, obj_total_vertices * sizeof(float), cudaMemcpyHostToDevice);
    test_collision_gpu(num_objects, num_collisions_gpu_broad, d_first_obj_index, d_second_obj_index, d_num_vertices,
                       d_vertices_coordinate_x, d_vertices_coordinate_y, d_vertices_coordinate_z, current_dim, d_collision_results, &gpu_time_narrow);
    collision_results = (bool *)malloc(num_collisions_gpu_broad * sizeof(bool));
    cudaMemcpy(collision_results, d_collision_results, num_collisions_gpu_broad * sizeof(bool), cudaMemcpyDeviceToHost);
    unsigned long real_collision_num = 0;
    // for (size_t i = 0; i < num_collisions_gpu_broad; i++) {
    //     if (collision_results[i]) {
    //         real_collision_num++;
    //     }
    // }

    // unsigned long num_objects = 2;
    // unsigned long num_collision = 1;
    // uint32_t first_obj_index[1] = {0};
    // uint32_t second_obj_index[1] = {1};
    // uint32_t num_vertices[2] = {6, 8};
    // float vertices_coordinate_x[14] = {4, 9, 4, 4, 9, 4, 6, 13, 11, 8, 6, 13, 11, 8};
    // float vertices_coordinate_y[14] = {11, 9, 5, 11, 9, 5, 7, 7, 2, 3, 7, 7, 2, 3};
    // float vertices_coordinate_z[14] = {11,11,11, 5, 5, 5, 4,4,4,4, 0, 0, 0, 0};
    // uint32_t * d_first_obj_index, * d_second_obj_index, * d_num_vertices;
    // float * d_vertices_coordinate_x, * d_vertices_coordinate_y, * d_vertices_coordinate_z;
    // bool * d_collision_results;
    // cudaMalloc((void **)&d_first_obj_index, sizeof(uint32_t));
    // cudaMalloc((void **)&d_second_obj_index, sizeof(uint32_t));
    // cudaMalloc((void **)&d_num_vertices, 2 * sizeof(uint32_t));
    // cudaMalloc((void **)&d_vertices_coordinate_x, 14 * sizeof(float));
    // cudaMalloc((void **)&d_vertices_coordinate_y, 14 * sizeof(float));
    // cudaMalloc((void **)&d_vertices_coordinate_z, 14 * sizeof(float));
    // cudaMalloc((void **)&d_collision_results, sizeof(bool));
    // cudaMemcpy(d_first_obj_index, first_obj_index, sizeof(uint32_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_second_obj_index, second_obj_index, sizeof(uint32_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_num_vertices, num_vertices, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_vertices_coordinate_x, vertices_coordinate_x, 14 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_vertices_coordinate_y, vertices_coordinate_y, 14 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_vertices_coordinate_z, vertices_coordinate_z, 14 * sizeof(float), cudaMemcpyHostToDevice);
    // test_collision_gpu(num_objects, num_collision, d_first_obj_index, d_second_obj_index, d_num_vertices,
    //                    d_vertices_coordinate_x, d_vertices_coordinate_y, d_vertices_coordinate_z, current_dim, d_collision_results);

    // thrust::device_ptr<bool> collision_results(d_collision_results);
    // std::cout << collision_results[0] << '\n';

    uint32_t * first_obj_index = (uint32_t *)malloc(num_collisions_gpu_broad * sizeof(uint32_t));
    uint32_t * second_obj_index = (uint32_t *)malloc(num_collisions_gpu_broad * sizeof(uint32_t));
    cudaMemcpy(first_obj_index, d_first_obj_index, num_collisions_gpu_broad * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(second_obj_index, d_second_obj_index, num_collisions_gpu_broad * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::unordered_set<std::pair<uint32_t, uint32_t>, pair_hashing, collision_pair_compare> collision_pairs;
    ofstream collision_pair_file("collision_pairs.txt");
    for (size_t i = 0; i < num_collisions_gpu_broad; i++) {
        auto result = collision_pairs.insert(std::make_pair(first_obj_index[i], second_obj_index[i]));
        if (result.second && collision_results[i]) {
            real_collision_num++;
            collision_pair_file << (*(result.first)).first << ' ' << (*(result.first)).second << '\n';
        }
    }
    collision_pair_file.close();
    std::cout << "Number of potential collisions detected by CPU = " << num_collisions_cpu_broad << " in " << cpu_time_broad << " ms." << '\n';
    std::cout << "Number of potential collisions detected by GPU = " << num_collisions_gpu_broad << " in " << gpu_time_broad << " ms." << '\n';
    std::cout << "Number of potential collisions detected by GPU = " << collision_pairs.size() << " (After de-duplicate)." << '\n';
    std::cout << "Number of real collisions detected by GPU = " << real_collision_num << " in " << gpu_time_narrow << " ms." << '\n';

    return 0;
}


float obj_init(float * coordinates, float * radii, unsigned long num_objects, unsigned int dim, int test_flag) {
    using namespace std;

    if (test_flag != 0) {
        ifstream testLayoutFile("testLayout.txt");
        if (testLayoutFile.is_open()) {
            for (size_t i = 0; i < num_objects; i++) {
                for (size_t j = 0; j < dim; j++) {
                    testLayoutFile >> coordinates[i + j * num_objects];
                }
                testLayoutFile >> radii[i];
            }
        }
    }
    else {
        // since later on we only use 8 bits to record the index of a cell in a dimension,
        // and the dimension of a cell is 1.5 * sqrt(2) * max(obj_radii),
        // so the entire space can be described by those 8 bits (in each dimension) is:
        // 256 * 1.5 * sqrt(2) * max(obj_radii), in order to guarantee objects bounding
        // sphere does not go to out of the entire space, we set it to:
        // 255 * 1.5 * sqrt(2) * max(obj_radii).
        // Basically, the allowed coordinates of objects are related to the size of the
        // biggest object. We cannot afford many tiny objects spared all over.
        srand(0);
        // let's say we want obj radii to be at most 5
        unsigned int allowed_max_radii = 5;
        for (size_t i = 0; i < num_objects; i++) {
            radii[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / allowed_max_radii));
        }
        float actual_max_radii = *std::max_element(radii, radii + num_objects);
        for (size_t i = 0; i < num_objects; i++) {
            for (size_t j = 0; j < dim; j++) {
                coordinates[i + j * num_objects] =
                    static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / actual_max_radii)) *
                    BOUNDING_SPHERE_SCALING * CELL_SIZE_SCALING * (MAX_CELL_NUM_ANY_DIM - 1) * 2;
                    // * 2 to turn radius to diameter
            }
        }
    }
    return *std::max_element(radii, radii + num_objects);
}

float bounding_sphere_init(float * obj_radii, float * bs_radii, unsigned long num_objects) {
    for (size_t i = 0; i < num_objects; i++) {
        bs_radii[i] = BOUNDING_SPHERE_SCALING * obj_radii[i];
    }
    return *std::max_element(bs_radii, bs_radii + num_objects);
}

void print_obj_info(float * obj_center_coordinates, float * obj_radii, unsigned int dim, unsigned long num_objects) {
    for (size_t i = 0; i < num_objects; i++) {
        for (size_t j = 0; j < dim; j++) {
            std::cout << obj_center_coordinates[i + j * num_objects] << '\t';
        }
        std::cout << obj_radii[i] << '\n';
    }
    std::cout << "Max radius = " << *std::max_element(obj_center_coordinates, obj_center_coordinates + dim * num_objects) << '\n';
}

void init_obj_index(unsigned int * obj_indices, unsigned int * obj_num_vertices, unsigned long num_objects, unsigned int available_obj_model) {
    using namespace std;
    // 1 obj = cube
    // 2 obj = cylinder
    // 3 obj = sphere
    // 4 obj = tetrahedron
    unsigned int model_vertices_num[4] = {114, 66, 8, 4};
    srand(0);

    for (size_t i = 0; i < num_objects; i++) {
        obj_indices[i] = rand() / (RAND_MAX / available_obj_model);
        obj_num_vertices[i] = model_vertices_num[obj_indices[i]];
    }
    // std::cout << "Max obj index = " << *std::max_element(obj_indices, obj_indices + num_objects) << '\n';
}


/*Split string with deliminator.*/
template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

/*Split the string to vector of strings.*/
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

/*Update the maximum and minimum values of objects.
 * */
void updateBoundary(float value, float *max, float *min) {
    if(value > *max) *max = value;
    if(value < *min) *min = value;
}

/*Perform the shift to positive direction for all the coordinates.*/
void shiftObject(double shiftDistance,double shiftDistance2,double shiftDistance3, int startIndex, int endIndex, float* x, float* y, float* z) {
     for(int i = startIndex; i < endIndex; i++) {
         x[i] += shiftDistance;
         y[i] += shiftDistance2;
         z[i] += shiftDistance3;
     }
}

/*Generate Random Number.*/
double randomNumber (double upperLimit) {
    return static_cast<float>(rand()) / (static_cast<float>(RAND_MAX)) * upperLimit;

    // //Random Number Generator for shift scalings.
    // std::random_device rd;  //Will be used to obtain a seed for the random number engine
    // std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    // std::uniform_real_distribution<> dis(0.0, upperLimit);
    // return dis(gen);
}

void readFile(float* xCoord, float* yCoord, float* zCoord, const std::string fileName, int dimension, int startIndex,
              float& xCentroid, float& yCentroid, float& zCentroid, float& objectLength) {
    std::ifstream in;
    in.open(fileName);
    if(dimension!=2 && dimension !=3) {
        std::cerr << "Need to be 2 or 3 dimension.";
        exit(1);
    }

    int counter = startIndex;
    float dimMax = 1;
    if (in.is_open()) {
        std::string line;
        float xMax=.0, yMax=.0, zMax=.0;
        float xMin=.0, yMin=.0, zMin=.0;

        //For any 3-dimensional objects.
        if(dimension == 3) {

            while (in >> line) {
                std::vector<std::string> tokens = split(line, ',');
                xCoord[counter] = (float) std::atof(tokens[0].c_str());
                updateBoundary(xCoord[counter], &xMax, &xMin);
                yCoord[counter] = (float) std::atof(tokens[1].c_str());
                updateBoundary(yCoord[counter], &yMax, &yMin);
                zCoord[counter] = (float) std::atof(tokens[2].c_str());
                updateBoundary(zCoord[counter], &zMax, &zMin);
                counter++;
            }
            dimMax = std::max(std::max(xMax - xMin, yMax - yMin), zMax - zMin);
        }
        //For any 2-dimensional object.
        else {
            while( in >> line) {
                std::vector<std::string> tokens = split(line,',');
                xCoord[counter] = (float) std::atof(tokens[0].c_str());
                updateBoundary(xCoord[counter], &xMax, &xMin);
                yCoord[counter] = (float) std::atof(tokens[1].c_str());
                updateBoundary(yCoord[counter], &yMax, &yMin);
                counter++;
            }
            dimMax = std::max(xMax - xMin,yMax - yMin);
        }
        //set the variables.
        objectLength = dimMax;
        xCentroid = (xMax + xMin)/2;
        yCentroid = (yMax + yMin)/2;
        zCentroid = (zMax + zMin)/2;
    }
}

void preProcess(float* x, float* y, float* z, float* Centroid,
                float* obj_radii, unsigned int * obj_indices, unsigned int num_objects, double space_scaling) {
    const int dim_3 = 3;
    const int start = 0;

    std::string cube = "cube.csv";
    float cubeXCentroid = 0.0;
    float cubeYCentroid = 0.0;
    float cubeZCentroid = 0.0;
    float cubeLength = 0.0;
    auto cubeX = (float*) malloc(sizeof(float)*8);
    auto cubeY = (float*) malloc(sizeof(float)*8);
    auto cubeZ = (float*) malloc(sizeof(float)*8);
    readFile(cubeX, cubeY, cubeZ, cube, dim_3, start,
             cubeXCentroid, cubeYCentroid, cubeZCentroid, cubeLength);

    std::string cylinder = "cylinder.csv";
    float cylinderXCentroid = 0.0;
    float cylinderYCentroid = 0.0;
    float cylinderZCentroid = 0.0;
    float cylinderLength = 0.0;
    auto cylinderX = (float*) malloc(sizeof(float)*66);
    auto cylinderY = (float*) malloc(sizeof(float)*66);
    auto cylinderZ = (float*) malloc(sizeof(float)*66);
    readFile(cylinderX, cylinderY, cylinderZ, cylinder, dim_3, start,
             cylinderXCentroid, cylinderYCentroid, cylinderZCentroid, cylinderLength);

    std::string sphere = "sphere.csv";
    float sphereXCentroid = 0.0;
    float sphereYCentroid = 0.0;
    float sphereZCentroid = 0.0;
    float sphereLength = 0.0;
    auto sphereX = (float*) malloc(sizeof(float)*114);
    auto sphereY = (float*) malloc(sizeof(float)*114);
    auto sphereZ = (float*) malloc(sizeof(float)*114);
    readFile(sphereX, sphereY, sphereZ, sphere, dim_3, start,
             sphereXCentroid, sphereYCentroid, sphereZCentroid, sphereLength);

    std::string tetrahedron = "tetrahedron.csv";
    float tetrahedronXCentroid = 0.0;
    float tetrahedronYCentroid = 0.0;
    float tetrahedronZCentroid = 0.0;
    float tetrahedronLength = 0.0;
    auto tetrahedronX = (float*) malloc(sizeof(float) *4);
    auto tetrahedronY = (float*) malloc(sizeof(float) *4);
    auto tetrahedronZ = (float*) malloc(sizeof(float) *4);
    readFile(tetrahedronX, tetrahedronY, tetrahedronZ, tetrahedron, dim_3, start,
             tetrahedronXCentroid, tetrahedronYCentroid, tetrahedronZCentroid, tetrahedronLength);

    double globalMax = CELL_SIZE_SCALING * BOUNDING_SPHERE_SCALING * (MAX_CELL_NUM_ANY_DIM - 1) *
            std::max(std::max(std::max(tetrahedronLength,cylinderLength),cubeLength),sphereLength);

    unsigned long counter=0;
    for(unsigned int i = 0; i < num_objects; i++) {
        int elementId = obj_indices[i];
        switch(elementId) {
            case 2: {
                //Allocate memory for object.
                float shiftDistance =  (float)randomNumber(space_scaling * globalMax);
                float shiftDistance2 =  (float)randomNumber(space_scaling * globalMax);
                float shiftDistance3 =  (float)randomNumber(space_scaling * globalMax);
                int size = 8;
                auto cubeXCopy = (float*) malloc(sizeof(float)*size);
                auto cubeYCopy = (float*) malloc(sizeof(float)*size);
                auto cubeZCopy = (float*) malloc(sizeof(float)*size);

                //Copy and shift.
                std::memcpy(cubeXCopy, cubeX, sizeof(float)*size);
                std::memcpy(cubeYCopy, cubeY, sizeof(float)*size);
                std::memcpy(cubeZCopy, cubeZ, sizeof(float)*size);
                shiftObject(shiftDistance, shiftDistance2, shiftDistance3,0, size, cubeXCopy, cubeYCopy, cubeZCopy);

                //Copy to main arrays.
                std::memcpy(x+counter,cubeXCopy, sizeof(float)*size);
                std::memcpy(y+counter,cubeYCopy, sizeof(float)*size);
                std::memcpy(z+counter,cubeZCopy, sizeof(float)*size);
                Centroid[i] = cubeXCentroid + shiftDistance;
                Centroid[num_objects + i] = cubeYCentroid + shiftDistance2;
                Centroid[2 * num_objects + i] = cubeZCentroid + shiftDistance3;
                obj_radii[i] = cubeLength / 2;

                //Free temporary arrays.
                free(cubeXCopy);
                free(cubeYCopy);
                free(cubeZCopy);
                counter += size;
            }break;
            case 1: {
                //Allocate memory for object.
                float shiftDistance =  (float)randomNumber(space_scaling * globalMax);
                float shiftDistance2 =  (float)randomNumber(space_scaling * globalMax);
                float shiftDistance3 =  (float)randomNumber(space_scaling * globalMax);
                int size = 66;
                auto cylinderXCopy = (float*) malloc(sizeof(float)*size);
                auto cylinderYCopy = (float*) malloc(sizeof(float)*size);
                auto cylinderZCopy = (float*) malloc(sizeof(float)*size);

                //Copy and shift.
                std::memcpy(cylinderXCopy, cylinderX, sizeof(float)*size);
                std::memcpy(cylinderYCopy, cylinderY, sizeof(float)*size);
                std::memcpy(cylinderZCopy, cylinderZ, sizeof(float)*size);
                shiftObject(shiftDistance, shiftDistance2, shiftDistance3, 0, size, cylinderXCopy, cylinderYCopy, cylinderZCopy);

                std::memcpy(x+counter,cylinderXCopy, sizeof(float)*size);
                std::memcpy(y+counter,cylinderYCopy, sizeof(float)*size);
                std::memcpy(z+counter,cylinderZCopy, sizeof(float)*size);
                Centroid[i]= cylinderXCentroid + shiftDistance;
                Centroid[num_objects + i]= cylinderYCentroid + shiftDistance2;
                Centroid[2 * num_objects + i] = cylinderZCentroid + shiftDistance3;
                obj_radii[i] = cylinderLength / 2;

                //Free temporary arrays.
                free(cylinderXCopy);
                free(cylinderYCopy);
                free(cylinderZCopy);
                counter += size;
            }break;
            case 0: {

                //Allocate memory for object.
                float shiftDistance =  (float)randomNumber(space_scaling * globalMax);
                float shiftDistance2 =  (float)randomNumber(space_scaling * globalMax);
                float shiftDistance3 =  (float)randomNumber(space_scaling * globalMax);
                int size = 114;
                auto sphereXCopy = (float*) malloc(sizeof(float)*size);
                auto sphereYCopy = (float*) malloc(sizeof(float)*size);
                auto sphereZCopy = (float*) malloc(sizeof(float)*size);

                //Copy and shift.
                std::memcpy(sphereXCopy, sphereX, sizeof(float)*size);
                std::memcpy(sphereYCopy, sphereY, sizeof(float)*size);
                std::memcpy(sphereZCopy, sphereZ, sizeof(float)*size);
                shiftObject(shiftDistance,shiftDistance2, shiftDistance3, 0, size, sphereXCopy, sphereYCopy, sphereZCopy);

                std::memcpy(x+counter,sphereXCopy, sizeof(float)*size);
                std::memcpy(y+counter,sphereYCopy, sizeof(float)*size);
                std::memcpy(z+counter,sphereZCopy, sizeof(float)*size);
                Centroid[i] = sphereXCentroid + shiftDistance;
                Centroid[num_objects + i] = sphereYCentroid + shiftDistance2;
                Centroid[2 * num_objects + i] = sphereZCentroid + shiftDistance3;
                obj_radii[i] = sphereLength / 2;

                //Free temporary arrays.
                free(sphereXCopy);
                free(sphereYCopy);
                free(sphereZCopy);
                counter += size;
            }
            break;
            case 3: {

                //Allocate memory for object.
                float shiftDistance =  (float)randomNumber(space_scaling * globalMax);
                float shiftDistance2 =  (float)randomNumber(space_scaling * globalMax);
                float shiftDistance3 =  (float)randomNumber(space_scaling * globalMax);
                int size = 4;
                auto tetrahedronXCopy = (float*) malloc(sizeof(float)*size);
                auto tetrahedronYCopy = (float*) malloc(sizeof(float)*size);
                auto tetrahedronZCopy = (float*) malloc(sizeof(float)*size);

                //Copy and shift.
                std::memcpy(tetrahedronXCopy, tetrahedronX, sizeof(float)*size);
                std::memcpy(tetrahedronYCopy, tetrahedronY, sizeof(float)*size);
                std::memcpy(tetrahedronZCopy, tetrahedronZ, sizeof(float)*size);
                shiftObject(shiftDistance, shiftDistance2, shiftDistance3,0, size, tetrahedronXCopy, tetrahedronYCopy, tetrahedronZCopy);

                std::memcpy(x+counter,tetrahedronXCopy, sizeof(float)*size);
                std::memcpy(y+counter,tetrahedronYCopy, sizeof(float)*size);
                std::memcpy(z+counter,tetrahedronZCopy, sizeof(float)*size);
                Centroid[i] = sphereXCentroid + shiftDistance;
                Centroid[num_objects + i] = sphereYCentroid + shiftDistance2;
                Centroid[2 * num_objects + i] = sphereZCentroid + shiftDistance3;
                obj_radii[i] = sphereLength / 2;
                counter += size;
            }break;
        }
    }
}
