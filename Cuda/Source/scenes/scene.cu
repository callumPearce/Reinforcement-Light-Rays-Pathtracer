#include "scene.cuh"

Scene::Scene(){

}

// Loads the cornell scene geometry into the scene
void Scene::load_cornell_box_scene(){

    // Load in the geometry into a vector dynamically
    std::vector<Surface> surfaces_v;
    std::vector<AreaLight> area_lights_v;
    get_cornell_shapes(surfaces_v, area_lights_v);

    // Set the size
    this->surfaces_count = surfaces_v.size();
    this->area_light_count = area_lights_v.size();

    //Convert to a fixed size array
    this->surfaces = new Surface[ this->surfaces_count ];
    this->area_lights = new AreaLight[ this->area_light_count ];
    memcpy(this->surfaces, &(surfaces_v[0]), sizeof(Surface) * this->surfaces_count);
    memcpy(this->area_lights, &(area_lights_v[0]), sizeof(AreaLight) * this->area_light_count);
}