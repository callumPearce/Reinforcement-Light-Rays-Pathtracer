#include "image_settings.h"
#include "shape.h"

Shape::Shape(Material material) {

    set_material(material);
}

//Getters
Material Shape::get_material(){
    return material;
}

//Setters
void Shape::set_material(Material material){
    this->material = material;
}