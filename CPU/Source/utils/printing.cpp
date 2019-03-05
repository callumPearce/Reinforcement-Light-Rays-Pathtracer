#include <iostream>
#include "printing.h"

void print_vec3(std::string name, vec3 v){
    std::cout << name << ": (" << v[0] << "," << v[1] << "," << v[2] << ")" << std::endl;
}

void print_vec4(std::string name, vec4 v){
    std::cout << name << ": (" << v[0] << "," << v[1] << "," << v[2] << "," << v[3] << ")" << std::endl;
}