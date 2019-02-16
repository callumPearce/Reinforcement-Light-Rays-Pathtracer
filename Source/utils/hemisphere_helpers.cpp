#include "hemisphere_helpers.h"

// Generate a random point on a unit hemisphere
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

// Create the transformation matrix for a unit hemisphere
mat4 create_transformation_matrix(vec3 normal, vec4 position){
    // Create coordinate system (i.e. 3 basis vectors to define rotation)
    vec3 normal_T;
    vec3 normal_B;
    create_normal_coordinate_system(normal, normal_T, normal_B);
    // Build the transformation matrix
    // [ right
    //   up
    //   forward
    //   translation ]
    vec4 normal4 = vec4(normal, 0.f);
    vec4 normal_T4 = vec4(normal_T, 0.f);
    vec4 normal_B4 = vec4(normal_B, 0.f);
    position.w = 1.f;
    return mat4(normal_T4, normal4, normal_B4, position);
}