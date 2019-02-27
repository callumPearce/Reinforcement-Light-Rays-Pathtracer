#include <stdio.h>
#include "printing.h"

void print_vec3(char name[], vec3 v){
    printf("%s : (%.4f. %.4f, %.4f)\n", name, v.x, v.y, v.z);
}

void print_vec4(char name[], vec4 v){
    printf("%s : (%.4f. %.4f, %.4f, %.4f)\n", name, v.x, v.y, v.z, v.w);
}