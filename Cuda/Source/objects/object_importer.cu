#include <iostream>
#include <algorithm>
#include "object_importer.cuh"
#include <cstring>

// Load a given object file into the scene to be rendered
__host__
bool load_scene(const char* path, std::vector<Surface>& surfaces, std::vector<AreaLight>& area_lights){

    // Attempt to open the object file in the supplied path
    FILE* file = fopen(path, "r");
    if (file == NULL){
        printf("File could not be opened!\n");
        return false;
    }

    // Data storage
    std::vector<vec3> vertex_indices;
    std::vector<vec3> temp_vertices;

    // Read the file until EOF
    while (1){
        char line_header[128];

        // Read the first word of the line
        int res = fscanf(file, "%s", line_header);

        // If the response is EOF we have finished
        if (res == EOF){
            break;
        } 
        // Else parse the obj file line
        else{
            // Vertex positions
            if (strcmp(line_header, "v") == 0){
                vec3 vertex;
                fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
                temp_vertices.push_back(vertex);
            }
            // Face of a traingles
            else if (strcmp(line_header, "f") == 0){

                // Get the line describing a face
                char face_line[256];
                if (fgets(face_line, 256, file) != NULL){

                    // Split the input up on spaces and append into args
                    std::vector<std::string> args;
                    std::string line = face_line;
                    split_string(args, line, " ");

                    // Further parse the x/x/x format
                    if (args.size() > 0 && args[0].find("/") != std::string::npos){

                        // For every args extract the first intege, this is the vertex dimension
                       std::vector<int> indices;
                        for (int i = 0; i < args.size(); i++){
                            std::vector<std::string> temp_args;
                            split_string(temp_args, args[i], "/");
                            indices.push_back(stoi(temp_args[0], nullptr, 10));
                        }
                        int start_index = indices[0];
                        for (int i = 1; i < indices.size()-1; i++){
                            int y = indices[i];
                            int z = indices[i+1];
                            vertex_indices.push_back(vec3(start_index,y,z));
                        }
                    } 
                    // Otherwise just convert the three integer indices (via Fan Triangulation)
                    else{
                        int start_vertex = stoi(args[0], nullptr, 10);
                        for (int i = 1; i < args.size()-1; i++){
                            int y = stoi(args[i], nullptr, 10);
                            int z = stoi(args[i+1], nullptr, 10);
                            vertex_indices.push_back(vec3(start_vertex,y,z));
                        }
                    }
                }
            }
        }
    }
    fclose (file);
    build_surfaces(surfaces, vertex_indices, temp_vertices);
    build_area_lights(area_lights);
}

// Convert the temporary stored data into triangles for rendering
__host__
void build_surfaces(std::vector<Surface>& surfaces, std::vector<vec3>& vertex_indices, std::vector<vec3>& temp_vertices){

    // Find the max and min vertex position of each dimension
    float max_pos[3] = {0.f};
    float min_pos[3] = {0.f};
    for (int i = 0; i < 3; i++){
        // Find the max and min
        for (int j = 0; j < temp_vertices.size(); j++){
            if (temp_vertices[j][i] > max_pos[i]){
                max_pos[i] = temp_vertices[j][i];
            }
            if (temp_vertices[j][i] < min_pos[i]){
                min_pos[i] = temp_vertices[j][i];
            }
        }
    }

    // Find the largest difference dimension
    float max_difference = 0.f;
    for (int i = 0; i < 3; i++){
        if (max_difference < std::fabs(max_pos[i] - min_pos[i])){
            max_difference = std::fabs(max_pos[i] - min_pos[i]);
        }
    }

    // Scale down/up to fit [-1,1] on max dimension
    float scale = 2.f;// / max_difference;
    
    // Translate so that a vertex can be at min [-1,-1,-1]
    float dist_x = -1.f -(min_pos[0] * scale);
    float dist_y = -1.f -(min_pos[1] * scale);
    float dist_z = -1.f -(min_pos[2] * scale);

    // For each vertex
    for (int i = 0; i < vertex_indices.size(); i++){

        // Get all the vertices and divide to scale
        vec4 v1 = vec4(temp_vertices[(int)vertex_indices[i].x - 1],1.f)*scale;
        vec4 v2 = vec4(temp_vertices[(int)vertex_indices[i].y - 1],1.f)*scale;
        vec4 v3 = vec4(temp_vertices[(int)vertex_indices[i].z - 1],1.f)*scale;
        v1 += vec4(dist_x, dist_y, dist_z, 0.f);
        v2 += vec4(dist_x, dist_y, dist_z, 0.f);
        v3 += vec4(dist_x, dist_y, dist_z, 0.f);
        
        vec4 rotation = vec4(-1.f, -1.f, 1.f, 1.f);

        v1 *= rotation;
        v2 *= rotation;
        v3 *= rotation;
        
        v1.w = 1.f;
        v2.w = 1.f;
        v3.w = 1.f;

        // Construct a Triangle with white material (TODO: Implement materials properly)
        // White
        Material mat = Material(vec3(0.75f));
        // Red
        if( i > 5 && i < 8){
            mat = Material(vec3(0.75f, 0.15f, 0.15f));
        }  
        // Blue
        // if( i % 5 == 0){
        //     mat = Material(vec3(0.15f, 0.15f, 0.75f));
        // }
        Surface surface = Surface(v1, v3, v2, mat);

        // Compute and set the normal
        surface.compute_and_set_normal();

        // Append to list of triangles
        surfaces.push_back(surface);
    }
}

// Given a std::string, seperate the std::string on a given delimiter and adds sub_strs to the std::vector
__host__
void split_string(std::vector<std::string>& sub_strs, std::string search_string, std::string delimiter){
    std::string token;
    size_t pos = 0;
    while ((pos = search_string.find(delimiter)) != std::string::npos) {
        // Get the substring
        token = search_string.substr(0, pos);
        
        // Check if there exists another delimter where we are not at the end of the std::string
        if (token != ""){
            // Convert to an int and add to list
            sub_strs.push_back(token);
        }

        // Delete the substring and the delimiter
        search_string.erase(0, pos + delimiter.length());
    }
    sub_strs.push_back(search_string);
}

// Builds some custom predifined area lights into the scene
__host__
void build_area_lights(std::vector<AreaLight>& area_lights){

    // Define the area light vectors
    float l = 2.f;
    vec4 I((7*l)/8, (7*l)/8, 2.f*l, 1);
    vec4 J((7*l)/8, l/8, 2.f*l, 1);
    vec4 K((4*l)/8, (7*l)/8, 2.f*l, 1);
    vec4 L((4*l)/8, l/8, 2.f*l, 1);

    vec3 diffuse_p = 2.f * vec3(0.9, 0.9, 0.9);
    AreaLight a1 = AreaLight(K, I, J, diffuse_p);
    area_lights.push_back(a1);

    AreaLight a2 = AreaLight(K, J, L, diffuse_p);
    area_lights.push_back(a2);

    // Resize the vertices to fit correctly into the scene
    for (size_t i = 0 ; i < area_lights.size() ; ++i) {
        area_lights[i].v0 = area_lights[i].v0 * (2 / l);
        area_lights[i].v1 = area_lights[i].v1 * (2 / l);
        area_lights[i].v2 = area_lights[i].v2 * (2 / l);

        area_lights[i].v0 = area_lights[i].v0 - vec4(1,1,1,1);
        area_lights[i].v1 = area_lights[i].v1 - vec4(1,1,1,1);
        area_lights[i].v2 = area_lights[i].v2 - vec4(1,1,1,1);

        vec4 newV0 = area_lights[i].v0;
        newV0.x *= -1;
        newV0.y *= -1;
        newV0.w = 1.0f;
        area_lights[i].v0 = newV0;

        vec4 newV1 = area_lights[i].v1;
        newV1.x *= -1;
        newV1.y *= -1;
        newV1.w = 1.0f;
        area_lights[i].v1 = newV1;

        vec4 newV2 = area_lights[i].v2;
        newV2.x *= -1;
        newV2.y *= -1;
        newV2.w = 1.0f;
        area_lights[i].v2 = newV2;

        area_lights[i].compute_and_set_normal();
    }
}