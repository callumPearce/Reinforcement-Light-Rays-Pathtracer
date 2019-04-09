#include "scene.cuh"

Scene::Scene(){

}

// Loads the cornell scene geometry into the scene
void Scene::load_cornell_box_scene(){

    // Load in the geometry into a vector dynamically
    std::vector<Surface> surfaces_v;
    std::vector<AreaLight> area_lights_v;
    std::vector<float> vertices_v;
    get_cornell_shapes(surfaces_v, area_lights_v, vertices_v);

    // Set the size
    this->surfaces_count = surfaces_v.size();
    this->area_light_count = area_lights_v.size();
    this->vertices_count = vertices_v.size();

    // Convert to a fixed size array
    this->surfaces = new Surface[ this->surfaces_count ];
    this->area_lights = new AreaLight[ this->area_light_count ];
    this->vertices = new float[ this->vertices_count ];
    memcpy(this->surfaces, &(surfaces_v[0]), sizeof(Surface) * this->surfaces_count);
    memcpy(this->area_lights, &(area_lights_v[0]), sizeof(AreaLight) * this->area_light_count);
    memcpy(this->vertices, &(vertices_v[0]), sizeof(float) * 3 * this->vertices_count);
}

// Loads a custom scene's geometry into the scene
void Scene::load_custom_scene(const char* filename){

    // Load the custom scene geometry into a vector dynamically
    std::vector<Surface> surfaces_v;
    std::vector<AreaLight> area_lights_v;
    std::vector<float> vertices_v;
    load_scene(filename, surfaces_v, area_lights_v, vertices_v);

    // Set the size
    this->surfaces_count = surfaces_v.size();
    this->area_light_count = area_lights_v.size();
    this->vertices_count = vertices_v.size();

    // Convert to a fixed size array
    this->surfaces = new Surface[ this->surfaces_count ];
    this->area_lights = new AreaLight[ this->area_light_count ];
    this->vertices = new float[ this->vertices_count * 3 ];
    memcpy(this->surfaces, &(surfaces_v[0]), sizeof(Surface) * this->surfaces_count);
    memcpy(this->area_lights, &(area_lights_v[0]), sizeof(AreaLight) * this->area_light_count);
    memcpy(this->vertices, &(vertices_v[0]), sizeof(float) * 3 * this->vertices_count);
}

// Save vertices to file
void Scene::save_vertices_to_file(){
    // Create the file 
    std::ofstream save_file ("../Radiance_Map_Data/vertices.txt");
    if (save_file.is_open()){

        for (int i = 0; i < this->surfaces_count; i++){
            Surface sf = this->surfaces[i];

            // Write the vertices
            save_file << sf.v0.x << " " << sf.v0.y << " " << sf.v0.z << " " << sf.v1.x << " " << sf.v1.y << " " << sf.v1.z << " " << sf.v2.x << " " << sf.v2.y << " " << sf.v2.z << "\n";
        }

        for (int i = 0; i < this->area_light_count; i++){
            AreaLight al = this->area_lights[i];

            // Write the vertices
            save_file << al.v0.x << " " << al.v0.y << " " << al.v0.z << " " << al.v1.x << " " << al.v1.y << " " << al.v1.z << " " << al.v2.x << " " << al.v2.y << " " << al.v2.z << "\n";
        }

        // Close the file
        save_file.close();
    }
    else{
        printf("Unable to save the vertices.\n");
    }
}