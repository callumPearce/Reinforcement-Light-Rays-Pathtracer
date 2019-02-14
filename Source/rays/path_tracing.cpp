#include "path_tracing.h"

vec3 path_trace(Ray ray, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes, int bounces){
    
    // Trace the path of the ray to find the closest intersection
    Intersection closest_intersection;
    ray.closest_intersection(surfaces, light_planes, closest_intersection);

    // Take the according action based on intersection type
    switch(closest_intersection.intersection_type){

        // Interescted with nothing, so no radiance
        case NOTHING:
            return vec3(0);
            break;
        
        // Intersected with light plane, so return its diffuse_p
        case AREA_LIGHT_PLANE:
            return light_planes[closest_intersection.index]->get_diffuse_p();
            break;

        // Intersected with a surface (diffuse)
        case SURFACE:
            if (bounces == MAX_RAY_BOUNCES){
                return vec3(0);
            } else{
                return indirect_radiance(closest_intersection, surfaces, light_planes, bounces);
            }
            break;
    }

    return vec3(0);
}

// For a given intersection point, return the radiance of the surface resulting
// from indirect illumination (i.e. other shapes in the scene) via the Monte Carlo Raytracing
vec3 indirect_radiance(const Intersection& intersection, vector<Surface *> surfaces, vector<AreaLightPlane *> light_planes, int bounces){

    // 1) Create new coordinate system (tranformation matrix)
    vec3 normal = vec3(intersection.normal.x, intersection.normal.y, intersection.normal.z);
    vec3 normal_T = vec3(0);
    vec3 normal_B = vec3(0);
    create_normal_coordinate_system(normal, normal_T, normal_B);

    // Calculate the estiamted total radiance estimate
    // (\int L(w_i) * roh / pi * cos(w_i, N) dw)
    vec3 total_radiance = vec3(0);
    for (int i = 0; i < SAMPLES_PER_BOUNCE; i++){
        
        // Generate random number for monte carlo sampling of theta and phi
        float cos_theta = ((float) rand() / (RAND_MAX)); //r1
        float r2 = ((float) rand() / (RAND_MAX));

        // 2) Sample uniformly coordinates on unit hemisphere
        vec3 sample = uniform_hemisphere_sample(cos_theta, r2);

        // 3) Transform random sampled position into the world coordinate system
        vec3 sampled_direction = vec3(
            sample.x * normal_B.x + sample.y * normal.x + sample.z * normal_T.x, 
            sample.x * normal_B.y + sample.y * normal.y + sample.z * normal_T.y, 
            sample.x * normal_B.z + sample.y * normal.z + sample.z * normal_T.z
        );

        // Create the new bounced ray
        vec4 start = intersection.position + (0.00001f * vec4(sampled_direction, 1));
        start[3] = 1.f;
        Ray ray = Ray(start, vec4(sampled_direction, 1));

        // 4) Get the radiance contribution for this ray and add to the sum
        vec3 radiance = path_trace(ray, surfaces, light_planes, bounces+1);

        // Note: we can multiply the BRDF to the final sum because it is 
        // constant for diffuse surfaces, so we omit it here
        total_radiance += cos_theta * radiance;
    }

    // Divide the sum by the number of samples (Monte Carlo) and apply BRDF
    // Note: 1/2pi comes from PDF being constant for sampled ray directions (Monte Carlo) 
    total_radiance /= ((float)SAMPLES_PER_BOUNCE * (1 / (2 * M_PI))); 
    total_radiance *= surfaces[intersection.index]->get_material().get_diffuse_c();

    return total_radiance;
}

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