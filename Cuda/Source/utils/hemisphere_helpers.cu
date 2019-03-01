#include "hemisphere_helpers.cuh"
//cuRand
#include <curand.h>
#include <curand_kernel.h>

// Generate a random point on a unit hemisphere
__device__
vec3 uniform_hemisphere_sample(float r1, float r2){

    // cos(theta) = 1 - r1 same as just doing r1
    float y = r1; 
    
    // theta = cos^-1 (1 - r1 - r1)
    float sin_theta = sqrt(1 - r1 * r1);

    // phi = 2*pi * r2
    float phi = 2 * M_PI * r2;

    // x = sin(theta) * cos(phi)
    float x = sin_theta * cosf(phi);
    // z = sin(theta) * sin(phi)
    float z = sin_theta * sinf(phi);

    return vec3(x,y,z);
}

// Create the new coordinate system based on the normal being the y-axis unit vector.
// In other words, create a **basis** set of vectors which any vector in the 3D space
// can be created with by taking a linear combination of these 3 vectors
__device__
void create_normal_coordinate_system(vec3& normal, vec3& normal_T, vec3& normal_B){
    // normal_T is found by setting either x or y to 0
    // i.e. the two define a plane
    if (fabs(normal.x) > fabs(normal.y)){
        // N_x * x = -N_z * z
        normal_T = normalize(vec3(normal.z, 0, -normal.x));
    } else{
        //N_y * y = -N_z * z
        normal_T = normalize(vec3(0, -normal.z, normal.y));
    }
    // The cross product between the two vectors creates another  
    // perpendicular to the plane formed by normal, normal_T
    normal_B = cross(normal, normal_T);
}

// // Create the transformation matrix for a unit hemisphere
// mat4 create_transformation_matrix(vec3 normal, vec4 position){
//     // Create coordinate system (i.e. 3 basis vectors to define rotation)
//     vec3 normal_T;
//     vec3 normal_B;
//     create_normal_coordinate_system(normal, normal_T, normal_B);
//     // Build the transformation matrix
//     // [ right
//     //   up
//     //   forward
//     //   translation ]
//     vec4 normal4 = vec4(normal, 0.f);
//     vec4 normal_T4 = vec4(normal_T, 0.f);
//     vec4 normal_B4 = vec4(normal_B, 0.f);
//     position.w = 1.f;
//     return mat4(normal_T4, normal4, normal_B4, position);
// }

// Sample a random direction in a unit hemisphere around an intersection point
__device__
vec4 sample_random_direction_around_intersection(curandState* d_rand_state, const Intersection& intersection, float& cos_theta){

    // Create new coordinate system (tranformation matrix)
    vec3 normal = vec3(intersection.normal.x, intersection.normal.y, intersection.normal.z);
    vec3 normal_T = vec3(0);
    vec3 normal_B = vec3(0);
    create_normal_coordinate_system(normal, normal_T, normal_B);
        
    // Generate random number for monte carlo sampling of theta and phi
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    curandState x_rand_state = d_rand_state[x*(int)SCREEN_HEIGHT + y];
    curandState y_rand_state = d_rand_state[x*(int)SCREEN_HEIGHT + y];
    cos_theta = curand_uniform(&x_rand_state);
    float r2 = curand_uniform(&y_rand_state);

    // Sample uniformly coordinates on unit hemisphere
    vec3 sample = uniform_hemisphere_sample(cos_theta, r2);

    // Transform random sampled direction into the world coordinate system
    vec3 sampled_direction = vec3(
        sample.x * normal_B.x + sample.y * normal.x + sample.z * normal_T.x, 
        sample.x * normal_B.y + sample.y * normal.y + sample.z * normal_T.y, 
        sample.x * normal_B.z + sample.y * normal.z + sample.z * normal_T.z
    );

    return vec4(sampled_direction, 1.f);
}