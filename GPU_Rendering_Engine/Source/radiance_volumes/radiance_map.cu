#include "radiance_map.cuh"
#include <algorithm>
#include <iostream>
#include <ctime>
#include "printing.h"

__host__
RadianceMap::RadianceMap(Surface* surfaces, int surfaces_count, std::vector<RadianceVolume>& temp_rvs, std::vector<RadianceTreeElement>& radiance_array_v){
    
    std::cout << "Sampling radiance volumes..." << std::endl;
    
    // Find the time
    time_t start_time;
    time_t end_time;
    time_t temp_time;
	start_time = time(NULL);

    // For every surface in the scene, uniformly sample N Radiance
    // Volumes where N is based on the triangles surface area
    get_radiance_volumes_count(surfaces, surfaces_count);
    size_t size = this->radiance_volumes_count * sizeof(RadianceVolume);
    size_t size_arr = this->radiance_array_size * sizeof(RadianceTreeElement);
    printf("RadianceMap size: %zu bytes\n",size);
    printf("RadianceArray size: %zu bytes\n",size_arr);
    temp_rvs =  std::vector<RadianceVolume>(this->radiance_volumes_count);
    uniformly_sample_radiance_volumes(surfaces, surfaces_count, temp_rvs);

    // Get a list of pointers
    std::vector<RadianceVolume*> temp_rvs_pointers;
    for (int i = 0; i < temp_rvs.size(); i++){
        temp_rvs_pointers.push_back(&(temp_rvs[i]));
        // printf("%d\n",temp_rvs_pointers[i]->index);
    }

    // Find the time
    end_time = time(NULL);
    temp_time = end_time - start_time; 
    std::cout << "Sampled " << temp_rvs.size() << " Radiance Volumes in " << temp_time << "s" << std::endl;

    // Create the RadianceTree (KDTree) from the radiance volumes
    start_time = end_time;
    std::cout << "Building Radiance Tree..." << std::endl; 

    RadianceTree* radiance_tree = new RadianceTree(temp_rvs_pointers, X_DIM);
    int radiance_array_s;

    radiance_tree->convert_to_array(radiance_array_s, radiance_array_v);
    RadianceTree::count_array_elements(radiance_array_v);
    this->radiance_array_size =radiance_array_s;

    end_time = time(NULL);
    temp_time = end_time - start_time;
    std::cout << "Radiance Tree constructed in " << temp_time << std::endl;
}


/*             Construction                */

__host__
void RadianceMap::get_radiance_volumes_count(Surface* surfaces, int surfaces_count){
    // Get sample count
    int total_sample_count = 0;
    for (int i = 0; i < surfaces_count; i++){
        total_sample_count += (int)floor(surfaces[i].compute_area() / AREA_PER_SAMPLE);
    }
    this->radiance_volumes_count = total_sample_count;
}

// Uniformly sample N Radiance volumes on a triangle based on the
// triangles surface area
__host__
void RadianceMap::uniformly_sample_radiance_volumes(Surface* surfaces, int surfaces_count, std::vector<RadianceVolume>& temp_rvs){
    int x = 0;
    for (int j = 0; j < surfaces_count; j++){
        // Calculate the number of radaince volumes to sample in that triangle
        int sample_count = (int)floor(surfaces[j].compute_area() / AREA_PER_SAMPLE);
        // Sample sample_count RadianceVolumes on the given surface
        for (int i = 0; i < sample_count; i++){
            vec4 sampled_position = surfaces[j].sample_position_on_plane();
            temp_rvs[x] = RadianceVolume(surfaces, sampled_position, surfaces[j].normal, j, x);
            x++;
        }
    }
}

/*              Querying                */

// Given an intersection point, importance sample a ray direction according to the
// cumulative distribution formed by the closest RadianceVolume's radiance_map
__device__
void RadianceMap::importance_sample_ray_direction(curandState* d_rand_state, const Intersection& intersection, int& sector_x, int& sector_y, int x, int y, vec4& sampled_direction, RadianceVolume* closest_volume, float& pdf){

    // If a radiance volume is not found, just sample randomly 
    if (closest_volume == NULL){
        float cos_theta;
        sampled_direction = sample_random_direction_around_intersection(d_rand_state, intersection.normal, cos_theta); 
        pdf = RHO;
    }
    else{
        // 2) Generate a random float uniformly between [0,1] and find which 
        //    part of the cumulative distribution this number falls in range
        //    of i.e. sample from the inverse of the cumulative distribution.
        //    This gives the location on the grid we sample our direction from
        sampled_direction = closest_volume->sample_direction_from_radiance_distribution(d_rand_state, x, y, sector_x, sector_y, pdf);
    }
}

// Performs the temporal difference update for the radiance volume for the sampled ray direction
// Follows https://arxiv.org/pdf/1701.07403.pdf update rule
__device__
RadianceVolume* RadianceMap::temporal_difference_update_radiance_volume_sector(float current_BRDF, RadianceVolume* current_radiance_volume, int current_sector_x, int current_sector_y, Intersection& intersection, Scene* scene){

    switch (intersection.intersection_type){

        case NOTHING:
            current_radiance_volume->temporal_difference_update(current_BRDF*ENVIRONMENT_LIGHT, current_sector_x, current_sector_y, scene->surfaces);
            return NULL;
            break;
        
        case AREA_LIGHT:
            float diffuse_light_power = scene->area_lights[intersection.index].luminance; 
            current_radiance_volume->temporal_difference_update(current_BRDF*diffuse_light_power, current_sector_x, current_sector_y, scene->surfaces);
            return NULL;
            break;
        
        case SURFACE:
            // Get the radiance volume closest to the intersection point
            RadianceVolume* closest_volume = this->find_closest_radiance_volume_iterative(MAX_DIST, intersection.position, intersection.normal);

            if (closest_volume == NULL){
                return NULL;
            }
            else{
                // Get the radiance incident from all directions for the next position and perform temporal diff update
                float next_pos_irradiance = closest_volume->get_irradiance_estimate(); 

                // if (closest_volume->irradiance_accum < 0.f)
                    // printf("%.3f\n",next_pos_irradiance);

                next_pos_irradiance *= current_BRDF;
                current_radiance_volume->temporal_difference_update(next_pos_irradiance, current_sector_x, current_sector_y, scene->surfaces);
                return closest_volume;
            }
            break;
    }
}

// Get the closest radiance volume iteratively
__device__
RadianceVolume* RadianceMap::find_closest_radiance_volume_iterative(float max_dist, vec4 pos, vec4 norm){

    vec3 position = vec3(pos);
    vec3 normal = vec3(norm);

    // return &(this->radiance_volumes[10]);

    // Intiliase the stack to keep track of tree search
    Stack stack = Stack(this->radiance_array_size);

    // Push the head of the tree index to begin the search
    stack.push(0);

    // Initialise to our search for the closest rv to the first rv
    int closest_rv_index = 0;
    float closest_dist = glm::distance(this->radiance_array[0].position, position);

    // Search the tree
    int index = 0;
    while(stack.pop(index)){

        // Check if the current index is a leaf and update accordingly
        RadianceTreeElement current_rte = this->radiance_array[index];
        if (current_rte.leaf){
            float dist = glm::distance(position, current_rte.position);
            if (normal == current_rte.normal && dist < closest_dist){
                closest_rv_index = current_rte.data;
                closest_dist = dist;
            }
        } 

        // Otherwise, search left or right
        else{
            float delta = position[current_rte.dimension] - this->radiance_array[index].data;
            if (delta < 0){
                if ( pow(delta,2) < max_dist){
                    stack.push(current_rte.right_idx);
                }
                // Left Branch
                stack.push(current_rte.left_idx);
            }
            else{
                if( pow(delta,2) < max_dist){
                    stack.push(current_rte.left_idx);
                }
                // Right Branch
                stack.push(current_rte.right_idx);
            }
        }
    }

    // return the found closest radiance volume
    return &(this->radiance_volumes[closest_rv_index]);
}

// Set the voronoi colours of all radiance volumes in the scene in the first entry of the radiance_grid[0][0]
__host__
void RadianceMap::set_voronoi_colours(std::vector<RadianceVolume>& temp_rvs){
    for (int i = 0; i < this->radiance_volumes_count; i++){
        temp_rvs[i].set_voronoi_colour();
    }
}

// Get the voronoi colour of the closest radiance volume
__device__
vec3 RadianceMap::get_voronoi_colour(const Intersection& intersection){
    // RadianceVolume* closest_volume = this->radiance_tree->find_closest_radiance_volume(MAX_DIST, intersection.position, intersection.normal);
    RadianceVolume* closest_volume = this->find_closest_radiance_volume_iterative(MAX_DIST, intersection.position, intersection.normal);
    vec4 pos = closest_volume->position;
    if (closest_volume != NULL){
        return closest_volume->get_voronoi_colour();   
    }
    else{
        return vec3(0.f);
    }
}

// Convert the radiance distributions from cumulative distribution to distribution
__host__
void RadianceMap::convert_radiance_volumes_distributions(){
    for (unsigned int i = 0; i < this->radiance_volumes_count; i++){
        this->radiance_volumes[i].convert_radiance_distribution();
    }
}

// Save the radiance map's q-values out to a file
__host__
void RadianceMap::save_q_vals_to_file(){

    // Create the file 
    std::ofstream save_file ("../Radiance_Map_Data/radiance_map_data.txt");
    if (save_file.is_open()){

        // Write the number of actions per radiance volume
        save_file << GRID_RESOLUTION * GRID_RESOLUTION << "\n";

        // Write each radiance volumes data
        for (int i = 0; i < this->radiance_volumes_count; i++){

            // Write the position
            vec4 position = this->radiance_volumes[i].position;
            save_file << position.x << " " << position.y << " " << position.z;

            // Write each Q values
            for (int n = 0; n < GRID_RESOLUTION*GRID_RESOLUTION; n++){
                save_file << " " << this->radiance_volumes[i].radiance_grid[n];
            }

            save_file << "\n";

        }

        // Close the file
        save_file.close();
    }
    else{
        printf("Unable to save the RadianceMap.\n");
    }
}