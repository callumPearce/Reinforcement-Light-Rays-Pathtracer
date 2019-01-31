#include <iostream>
#include "printing.h"

void print_vec3(string name, vec3 v){
    cout << name << ": (" << v[0] << "," << v[1] << "," << v[2] << ")" << endl;
}

void print_vec4(string name, vec4 v){
    cout << name << ": (" << v[0] << "," << v[1] << "," << v[2] << "," << v[3] << ")" << endl;
}