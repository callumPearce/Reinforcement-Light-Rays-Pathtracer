#ifndef RADIANCE_VOLUME_DATA_EXTRACTOR_H
#define RADIANCE_VOLUME_DATA_EXTRACTOR_H

/* Project */
#include "radiance_volumes_settings.h"
#include "hemisphere_helpers.cuh"
#include "image_settings.h"
#include "dq_network.cuh"
#include "scene.cuh"

/* Dynet */
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

// File writing
#include <iostream>
#include <fstream>

/* GLM */
#include <glm/glm.hpp>
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

// Save the selected radiance volume data specified in to_select.txt
// located in the file path passed in
void save_selected_radiance_volumes_vals_nn(
    std::string data_fpath, 
    std::string network_fname,
    Scene& scene,
    int argc,
    char** argv
);


#endif