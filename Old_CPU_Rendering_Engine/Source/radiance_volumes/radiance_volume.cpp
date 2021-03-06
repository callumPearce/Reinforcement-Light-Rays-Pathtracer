#include "radiance_volume.h"
#include <iostream>

RadianceVolume::RadianceVolume(){
    this->position = vec4(0.f);
    this->normal = vec3(0.f);
    this->transformation_matrix = mat4(0.f);
}

RadianceVolume::RadianceVolume(vec4 position, vec4 normal){
    initialise_radiance_grid();
    initialise_radiance_distribution();
    initialise_visits();
    set_position(position);
    set_normal(vec3(normal.x, normal.y, normal.z));

    // Create the transformation matrix for this hemisphere: local->world
    this->transformation_matrix = create_transformation_matrix(normal, position);
}

// Updates the transformation matrix with the current set values of the normal and position
void RadianceVolume::update_transformation_matrix(){
    this->transformation_matrix = create_transformation_matrix(normal, position);
} 

// Intialises a 2D grid to store radiance values at each grid point
void RadianceVolume::initialise_radiance_grid(){
    for (int x = 0; x < GRID_RESOLUTION; x++){
        std::vector<vec3> grid_row;
        for (int y = 0; y < GRID_RESOLUTION; y++){
            grid_row.push_back(vec3(1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION)));
        }
        this->radiance_grid.push_back(grid_row);
    }
}

// Initialise radiance distribution to be equal in all angles initially
void RadianceVolume::initialise_radiance_distribution(){
    for (int x = 0; x < GRID_RESOLUTION; x++){
        std::vector<float> distribution_row;
        for (int y = 0; y < GRID_RESOLUTION; y++){
            distribution_row.push_back(1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION));
        }
        this->radiance_distribution.push_back(distribution_row);
    }
}

// Initialises the alpha values (weighting of state-action pairs) to be 1
void RadianceVolume::initialise_visits(){
    for (int x = 0; x < GRID_RESOLUTION; x++){
        std::vector<float> visits_row;
        for (int y = 0; y < GRID_RESOLUTION; y++){
            visits_row.push_back(0.f);
        }
        this->visits.push_back(visits_row);
    }
}

// Gets the incoming radiance values from all grid samples and
// populates radiance_grid with the estimates
// NOTE: This should be called before any radiance_volumes are instantiated
// in the scene by surfaces or these surfaces will be taken into account
void RadianceVolume::get_radiance_estimate(std::vector<Surface *> surfaces, std::vector<AreaLightPlane *> light_planes){
    // Path trace a ray in the direction from the centre of hemisphere towards
    // the centre of each sector to determine the radiance incoming from that direction
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            // Path trace SAMPLES_PER_PIXEL rays and average their value
            vec3 radiance = vec3(0);
            for (int i = 0; i < SAMPLES_PER_PIXEL; i++){
                // Get the 3D coordinates of a random point in the sector
                float r1 = ((float) rand() / (RAND_MAX));
                float r2 = ((float) rand() / (RAND_MAX));
                float sector_x, sector_y, sector_z;
                map(
                    (x + r1)/(float)GRID_RESOLUTION,
                    (y + r2)/(float)GRID_RESOLUTION,
                    sector_x, 
                    sector_y, 
                    sector_z
                );
                sector_x *= DIAMETER;
                sector_y *= DIAMETER;
                sector_z *= DIAMETER;
                // Convert the position to world space
                vec4 world_position = this->transformation_matrix * vec4(sector_x, sector_y, sector_z, 1.f);
                // Calculate the direction and starting position of the ray
                vec4 dir = world_position - vec4(this->position.x, this->position.y, this->position.z, 0.f);
                vec4 start = this->position + 0.00001f * dir;
                start.w = 1.f;
                // Create the ray and path trace to find the radiance in that direction
                Ray ray = Ray(start, dir);
                radiance += path_trace_recursive(ray, surfaces, light_planes, 0);
            }
            this->radiance_grid[x][y] = radiance / (float)SAMPLES_PER_PIXEL;
        }
    }
}

// Builds a radiance volume out of Surfaces, where each surfaces colour
// represents the incoming radiance at that position from that angle
void RadianceVolume::build_radiance_volume_shapes(std::vector<Surface>& surfaces){
    // Get its vertices
    std::vector<std::vector<vec4>> vertices;
    this->get_vertices(vertices);
    // Build Surfaces using the vertices
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            vec4 v1 = vertices[x][y];
            vec4 v2 = vertices[x+1][y];
            vec4 v3 = vertices[x][y+1];
            vec4 v4 = vertices[x+1][y+1];
            float r1 = ((float) rand() / (RAND_MAX));
            float r2 = ((float) rand() / (RAND_MAX));
            float r3 = ((float) rand() / (RAND_MAX));
            surfaces.push_back(Surface(v1, v3, v2, Material(this->radiance_grid[x][y])));
            surfaces.push_back(Surface(v2, v3, v4, Material(this->radiance_grid[x][y])));
        }
    }
}

// Builds a radiance volume out of Surfaces, where each surfaces colour
// represents the incoming radiance at that position from that angle
void RadianceVolume::build_radiance_magnitude_volume_shapes(std::vector<Surface>& surfaces){
    // Get its vertices
    std::vector<std::vector<vec4>> vertices;
    this->get_vertices(vertices);
    // Find the max radiance magnitude of the hemisphere
    float max_radiance = 0.0001f;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            if(length(this->radiance_grid[x][y]) > max_radiance){
                max_radiance = length(this->radiance_grid[x][y]);
            }
        }
    }
    // Build Surfaces using the vertices
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            vec4 v1 = vertices[x][y];
            vec4 v2 = vertices[x+1][y];
            vec4 v3 = vertices[x][y+1];
            vec4 v4 = vertices[x+1][y+1];
            float r1 = ((float) rand() / (RAND_MAX));
            float r2 = ((float) rand() / (RAND_MAX));
            float r3 = ((float) rand() / (RAND_MAX));
            vec3 colour = vec3(1.f) - length(this->radiance_grid[x][y])/max_radiance * vec3(1.f);
            surfaces.push_back(Surface(v1, v3, v2, Material(colour)));
            surfaces.push_back(Surface(v2, v3, v4, Material(colour)));
        }
    }
}

// Returns a list of vertices for the generated radiance volume
void RadianceVolume::get_vertices(std::vector<std::vector<vec4>>& vertices){
    // For every grid coordinate, add the corresponding 3D world coordinate
    for (int x = 0; x <= GRID_RESOLUTION; x++){
        std::vector<vec4> vertices_row;
        for (int y = 0; y <= GRID_RESOLUTION; y++){
            // Get the coordinates on the unit hemisphere
            float x_h, y_h, z_h;
            map(x/(float)GRID_RESOLUTION, y/(float)GRID_RESOLUTION, x_h, y_h, z_h);
            // Scale to the correct diameter desired of the hemisphere
            x_h *= DIAMETER;
            y_h *= DIAMETER;
            z_h *= DIAMETER;
            // Convert to world space
            vec4 world_position = this->transformation_matrix * vec4(x_h, y_h, z_h, 1.f);
            // Add the point to vertices_row
            vertices_row.push_back(world_position);
        }
        vertices.push_back(vertices_row);
    }
}

// Gets the irradiance for an intersection point by solving the rendering equations (summing up 
// radiance from all directions whilst multiplying by BRDF and cos(theta))
vec3 RadianceVolume::get_irradiance(const Intersection& intersection, std::vector<Surface *> surfaces){
    vec3 irradiance = vec3(0);
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            // Get the coordinates on the unit hemisphere
            float x_h, y_h, z_h;
            map(x/(float)GRID_RESOLUTION, y/(float)GRID_RESOLUTION, x_h, y_h, z_h);
            // Scale to the correct diameter desired of the hemisphere
            x_h *= DIAMETER;
            y_h *= DIAMETER;
            z_h *= DIAMETER;
            // Convert to world space
            vec4 world_position = this->transformation_matrix * vec4(x_h, y_h, z_h, 1.f);
            vec3 world_position3 = vec3(world_position.x, world_position.y, world_position.z);
            // Get the direction
            vec3 dir = normalize(world_position3 - vec3(this->position));
            // Get the angle between the dir std::vector and the normal
            float cos_theta = dot(dir, this->normal); // No need to divide by lengths as they have been normalized
            irradiance += cos_theta * this->radiance_grid[x][y];
        }
    }
    irradiance /= ((float)(GRID_RESOLUTION * GRID_RESOLUTION)) * (1.f / (2.f * M_PI));
    irradiance *= surfaces[intersection.index]->get_material().get_diffuse_c();
    return irradiance;
}

// Normalizes this RadianceVolume so that all radiance values 
// i.e. their grid values all sum to 1 (taking the length of each vec3)
void RadianceVolume::update_radiance_distribution(){
    // Get the total radiance from all directions (as a float)
    vec3 total_rgb = vec3(0.f);
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            total_rgb += this->radiance_grid[x][y];
        }
    }
    // Use this total to convert all radiance_grid values into probabilities
    // and store in the radiance_distribution
    float total = length(total_rgb);
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            this->radiance_distribution[x][y] = length(this->radiance_grid[x][y])/total;
        }
    }
}

// Samples a direction from the radiance_disset_voronoi_colours()
// volume
vec4 RadianceVolume::sample_direction_from_radiance_distribution(int& sector_x, int& sector_y){
    
    // Generate a random float uniformly betset_voronoi_colours()
    float r = ((float) rand() / (RAND_MAX));

    // Sample from the inverse cumulative distribution
    float cumulative_sum = 0.f;
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            cumulative_sum += this->radiance_distribution[x][y];
            // We have found where in the inverse cumulative distribution our
            // sample is. There for return an anlge at this location
            if ( r <= cumulative_sum ){
                sector_x = x;
                sector_y = y;
                // Get the coordinates on the unit hemisphere
                float x_h, y_h, z_h;
                map(x/(float)GRID_RESOLUTION, y/(float)GRID_RESOLUTION, x_h, y_h, z_h);
                // Convert to world space
                vec4 world_position = this->transformation_matrix * vec4(x_h, y_h, z_h, 1.f);
                vec3 world_position3 = vec3(world_position.x, world_position.y, world_position.z);
                // Get the direction
                return vec4(normalize(world_position3 - vec3(this->position)),1.f);
            }
        }
    }
}

// Performs a temporal difference update for the current radiance volume for the incident
// radiance in the sector specified with the intersection surfaces irradiance value
void RadianceVolume::temporal_difference_update(vec3 next_irradiance, int sector_x, int sector_y){
    #pragma omp critical
    {
        // Calculate alpha and update the radiance grid values and increment the number of visits
        float alpha = 1.f / (1.f + this->visits[sector_x][sector_y]);
        this->radiance_grid[sector_x][sector_y] = ((1.f - (alpha)) * this->radiance_grid[sector_x][sector_y]) + (alpha * next_irradiance);
        this->visits[sector_x][sector_y] += 1;
    }
}

// Sets a voronoi colour for the radiance volume (random colour) in the first entry of its radiance grid
void RadianceVolume::set_voronoi_colour(){
    float r = ((float) rand() / (RAND_MAX));
    float g = ((float) rand() / (RAND_MAX));
    float b = ((float) rand() / (RAND_MAX));
    this->radiance_grid[0][0] = vec3(r,g,b);
}

// Gets the voronoi colour of the radiance volume
vec3 RadianceVolume::get_voronoi_colour(){
    return this->radiance_grid[0][0];
}

/*
* This function takes a point in the unit square,
* and maps it to a point on the unit hemisphere.
*
* Copyright 1994 Kenneth Chiu
*
* This code may be freely distributed and used
* for any purpose, commercial or non-commercial,
* as long as attribution is maintained.
*/
void RadianceVolume::map(float x, float y, float& x_ret, float& y_ret, float& z_ret) {
    float xx, yy, offset, theta, phi;
    x = 2*x - 1;
    y = 2*y - 1;
    if (y > -x) { // Above y = -x
        if (y < x) { // Below y = x
            xx = x;
            if (y > 0) { // Above x-axis
                /*
                * Octant 1
                */
                offset = 0;
                yy = y;
            } 
            else { // Below and including x-axis
                /*
                * Octant 8
                */
                offset = (7*M_PI)/4;
                yy = x + y;
            }
        } 
        else { // Above and including y = x
            xx = y;
            if (x > 0) { // Right of y-axis
                /*
                * Octant 2
                */
                offset = M_PI/4;
                yy = (y - x);
            } 
            else { // Left of and including y-axis
                /*
                * Octant 3
                */
                offset = (2*M_PI)/4;
                yy = -x;
            }
        }
    } 
    else { // Below and including y = -x
        if (y > x) { // Above y = x
            xx = -x;
            if (y > 0) { // Above x-axis
                /*
                * Octant 4
                */
                offset = (3*M_PI)/4;
                yy = -x - y;
            } 
            else { // Below and including x-axis
                /*
                * Octant 5
                */
                offset = (4*M_PI)/4;
                yy = -y;
            }
        } 
        else { // Below and including y = x
            xx = -y;
            if (x > 0) { // Right of y-axis
                /*
                * Octant 7
                */
                offset = (6*M_PI)/4;
                yy = x;
            } 
            else { // Left of and including y-axis
                if (y != 0) {
                    /*
                    * Octant 6
                    */
                    offset = (5*M_PI)/4;
                    yy = x - y;
                } 
                else {
                    /*
                    * Origincreate_normal_coordinate_system
                    */
                    x_ret = 0.f;
                    y_ret = 1.f;
                    z_ret = 0.f;
                    return;
                }
            }
        }
    }
    theta = acos(1 - xx*xx);
    phi = offset + (M_PI/4)*(yy/xx);
    x_ret = sin(theta)*cos(phi);
    y_ret = cos(theta);
    z_ret = sin(theta)*sin(phi);
}

// Getters
vec4 RadianceVolume::get_position(){
    return this->position;
}

vec3 RadianceVolume::get_normal(){
    return this->normal;
}

// Setters
void RadianceVolume::set_position(vec4 position){
    this->position = position;
}

void RadianceVolume::set_normal(vec3 normal){
    this->normal = normal;
}