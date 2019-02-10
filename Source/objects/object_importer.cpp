#include <iostream>
#include <algorithm>
#include "object_importer.h"

// Load a given object file into the scene to be rendered
bool load_scene(const char* path, vector<Triangle>& triangles){

    // Attempt to open the object file in the supplied path
    FILE* file = fopen(path, "r");
    if (file == NULL){
        printf("File could not be opened!\n");
        return false;
    }

    // Data storage
    vector<vec3> vertex_indices;
    vector<vec3> temp_vertices;

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
                    vector<string> args;
                    string line = face_line;
                    split_string(args, line, " ");

                    // Further parse the x/x/x format
                    if (args.size() > 0 && args[0].find("/") != string::npos){

                        // For every args extract the first intege, this is the vertex dimension
                       vector<int> indices;
                        for (int i = 0; i < args.size(); i++){
                            vector<string> temp_args;
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
    build_triangles(triangles, vertex_indices, temp_vertices);
}

// Convert the temporary stored data into triangles for rendering
void build_triangles(vector<Triangle>& triangles, vector<vec3>& vertex_indices, vector<vec3>& temp_vertices){

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
        if (max_difference < max_pos[i] - min_pos[i]){
            max_difference = max_pos[i] - min_pos[i];
        }
    }

    // Scale down/up to fit [-1,1] on max dimension
    float scale = 2 / max_difference;
    
    // Translate so that a vertex can be at min [-1,-1,-1]
    // float dist_x = -1.f;//(min_pos[0] * scale);
    // float dist_y = -1.f;//(min_pos[1] * scale);
    // float dist_z = -1.f;//(min_pos[2] * scale);

    // For each vertex
    for (int i = 0; i < vertex_indices.size(); i++){

        // Get all the vertices and divide to scale
        vec4 v1 = vec4(temp_vertices[(int)vertex_indices[i].x - 1],1.f)*scale;
        vec4 v2 = vec4(temp_vertices[(int)vertex_indices[i].y - 1],1.f)*scale;
        vec4 v3 = vec4(temp_vertices[(int)vertex_indices[i].z - 1],1.f)*scale;
        // v1 += vec4(dist_x, dist_y, dist_z, 0.f);
        // v2 += vec4(dist_x, dist_y, dist_z, 0.f);
        // v3 += vec4(dist_x, dist_y, dist_z, 0.f);
        
        vec4 rotation = vec4(-1.f, -1.f, 1.f, 1.f);

        v1 *= rotation;
        v2 *= rotation;
        v3 *= rotation;
        
        v1.w = 1.f;
        v2.w = 1.f;
        v3.w = 1.f;

        // Construct a Triangle with white material (TODO: Implement materials properly)
        Triangle triangle = Triangle(v1, v3, v2, Material(vec3(0.75f, 0.75f, 0.75f)));

        // Compute and set the normal
        triangle.compute_and_set_normal();

        // Append to list of triangles
        triangles.push_back(triangle);
    }
}

// Given a string, seperate the string on a given delimiter and adds sub_strs to the vector
void split_string(vector<string>& sub_strs, string search_string, string delimiter){
    string token;
    size_t pos = 0;
    while ((pos = search_string.find(delimiter)) != string::npos) {
        // Get the substring
        token = search_string.substr(0, pos);
        
        // Check if there exists another delimter where we are not at the end of the string
        if (token != ""){
            // Convert to an int and add to list
            sub_strs.push_back(token);
        }

        // Delete the substring and the delimiter
        search_string.erase(0, pos + delimiter.length());
    }
    sub_strs.push_back(search_string);
}