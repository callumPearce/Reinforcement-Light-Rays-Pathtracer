#include <iostream>
#include <algorithm>
#include "object_importer.cuh"
#include <cstring>

// Load a given object file into the scene to be rendered
__host__
bool load_scene(const char* path, std::vector<Surface>& surfaces, std::vector<AreaLight>& area_lights, std::vector<float>& vertices, bool lights_in_obj){

    // Attempt to open the object file in the supplied path
    FILE* file = fopen(path, "r");
    if (file == NULL){
        printf("File %s could not be opened!\n", path);
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
    if(lights_in_obj){
        build_surfaces_and_lights(surfaces, area_lights, vertices, vertex_indices, temp_vertices);
    }else{
        build_surfaces(surfaces, vertices, vertex_indices, temp_vertices);
        build_area_lights(area_lights, vertices);
    }
}

// Convert the temporary stored data into triangles for rendering
__host__
void build_surfaces(std::vector<Surface>& surfaces, std::vector<float>& vertices, std::vector<vec3>& vertex_indices, std::vector<vec3>& temp_vertices){

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
                
        // Red (FOR DOOR SCENE)
        // if( i > 23 && i < 36){
            // mat = Material(vec3(0.75f, 0.15f, 0.15f));
        // }  
        // Red (FOR ARCHWAY SCENE)
        if(i > 80){
            mat = Material(vec3(0.75f, 0.15f, 0.15f));
        }
        // Blue 
        if( 11 < i && i < 24){
            mat = Material(vec3(0.15f, 0.15f, 0.75f));
        }

        // Create the surface
        Surface surface = Surface(v1, v3, v2, mat);
        
        // Compute and set the normal
        surface.compute_and_set_normal();

        // Append to list of triangles
        surfaces.push_back(surface);

        // Add the vertices to the vector
        vertices.push_back(v1.x);
        vertices.push_back(v1.y);
        vertices.push_back(v1.z);
        vertices.push_back(v2.x);
        vertices.push_back(v2.y);
        vertices.push_back(v2.z);
        vertices.push_back(v3.x);
        vertices.push_back(v3.y);
        vertices.push_back(v3.z);
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
void build_area_lights(std::vector<AreaLight>& area_lights, std::vector<float>& vertices){
    
    vec3 diffuse_p = 8.f * vec3(1.f, 1.f, 1.f);

    // Define the area light vectors ( Door room )
    float l = 2.f;
    // vec4 I((6.3f*l)/8, (l*6.f)/8, 1.499f*l, 1);
    // vec4 J((6.3f*l)/8, 0, 1.499f*l, 1);
    // vec4 K((2.58f*l)/8, (l*6.f)/8, 1.499f*l, 1);
    // vec4 L((2.58f*l)/8, 0, 1.499f*l, 1);

    // // ( Simple Closed Room )
    // // vec4 I(l - 0.001f, (l*4.f)/8, 1.f*l, 1.f);
    // // vec4 J(l - 0.001f, (l*1.f)/8, 1.f*l, 1.f);
    // // vec4 K(l - 0.001f, (l*4.f)/8, 0.5f*l, 1.f);
    // // vec4 L(l - 0.001f, (l*1.f)/8, 0.5f*l, 1.f);

    // // ( Simple Room )
    // // vec4 I(l - 0.001f, (l*6.f)/8, 0.5f*l, 1.f);
    // // vec4 J(l - 0.001f, (l*3.f)/8, 0.5f*l, 1.f);
    // // vec4 K(l - 0.001f, (l*6.f)/8, 0.25f*l, 1.f);
    // // vec4 L(l - 0.001f, (l*3.f)/8, 0.25f*l, 1.f);

    // AreaLight a1 = AreaLight(K, I, J, diffuse_p);
    // area_lights.push_back(a1);

    // AreaLight a2 = AreaLight(K, J, L, diffuse_p);
    // area_lights.push_back(a2);

    // ( Archway )
    vec4 I(l+ 1.99f, l, 2.5*l, 1.f);
    vec4 J(l+ 1.99f, (l*4.f)/8, 2.5f*l, 1.f);
    vec4 K(l+ 1.99f, l, 2.f*l, 1.f);
    vec4 L(l+ 1.99f, (l*4.f)/8, 2.f*l, 1.f);

    AreaLight a1 = AreaLight(K, I, J, diffuse_p);
    area_lights.push_back(a1);

    AreaLight a2 = AreaLight(K, J, L, diffuse_p);
    area_lights.push_back(a2);

    vec4 M(l- 1.99f, l, 2.5f*l, 1.f);
    vec4 N(l- 1.99f, (l*4.f)/8, 2.5f*l, 1.f);
    vec4 O(l- 1.99f, l, 2.0f*l, 1.f);
    vec4 P(l- 1.99f, (l*4.f)/8, 2.0f*l, 1.f);

    AreaLight a3 = AreaLight(O, M, N, diffuse_p);
    area_lights.push_back(a3);

    AreaLight a4 = AreaLight(O, N, P, diffuse_p);
    area_lights.push_back(a4);

    vec4 Q(l- 0.5f, l, 2.99f*l, 1.f);
    vec4 R(l- 0.5f, l * 0.5f, 2.99f*l, 1.f);
    vec4 S(l+ 0.5f, l, 2.99f*l, 1.f);
    vec4 T(l+ 0.5f, l * 0.5f, 2.99f*l, 1.f);

    AreaLight a5 = AreaLight(S, Q, R, diffuse_p);
    area_lights.push_back(a5);

    AreaLight a6 = AreaLight(S, R, T, diffuse_p);
    area_lights.push_back(a6);

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

        // Add to vertices list
        vertices.push_back(newV0.x);
        vertices.push_back(newV0.y);
        vertices.push_back(newV0.z);
        vertices.push_back(newV1.x);
        vertices.push_back(newV1.y);
        vertices.push_back(newV1.z);
        vertices.push_back(newV2.x);
        vertices.push_back(newV2.y);
        vertices.push_back(newV2.z);

        area_lights[i].compute_and_set_normal();
    }
}

// Convert the temporary stored data into triangles for rendering
__host__
void build_surfaces_and_lights(std::vector<Surface>& surfaces, std::vector<AreaLight>& area_lights, std::vector<float>& vertices, std::vector<vec3>& vertex_indices, std::vector<vec3>& temp_vertices){

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
        
        // Lights
        if( (i > 23 && i < 36) || (i > 50 && i < 63)){
            vec3 diffuse_p = 12.f * vec3(1.f, 1.f, 1.f);
            AreaLight a1 = AreaLight(v1, v3, v2, diffuse_p);
            area_lights.push_back(a1);
        }  
        else{
            // White
            Material mat = Material(vec3(0.9f));

            if ((i >= 0 && i <= 7) ){
                mat = Material(vec3(0.1f));
            }
            else if ((i > 133 && i < 142)){
                mat = Material(vec3(0.75f, 0.15f, 0.15f));
            }

            // Create the surface
            Surface surface = Surface(v1, v3, v2, mat);
            
            // Compute and set the normal
            surface.compute_and_set_normal();

            // Append to list of triangles
            surfaces.push_back(surface);
        }

        // Add the vertices to the vector
        vertices.push_back(v1.x);
        vertices.push_back(v1.y);
        vertices.push_back(v1.z);
        vertices.push_back(v2.x);
        vertices.push_back(v2.y);
        vertices.push_back(v2.z);
        vertices.push_back(v3.x);
        vertices.push_back(v3.y);
        vertices.push_back(v3.z);
    }
}
