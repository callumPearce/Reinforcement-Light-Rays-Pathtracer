#include "object_importer.h"

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
                string vertex_1, vertex_2, vertex_3;
                unsigned int vertex_index[3], uv_index[3], normal_index[3];
                int matches = fscanf
                (
                    file, 
                    "%d/%d/%d %d/%d/%d %d/%d/%d\n", 
                    &vertex_index[0], 
                    &uv_index[0], 
                    &normal_index[0], 
                    &vertex_index[1], 
                    &uv_index[1], 
                    &normal_index[1], 
                    &vertex_index[2], 
                    &uv_index[2], 
                    &normal_index[2] 
                );
                if (matches != 9){
                    printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                    return false;
                }
                vertex_indices.push_back(vec3(vertex_index[0], vertex_index[1], vertex_index[2]));
            }
        }
    }

    build_triangles(triangles, vertex_indices, temp_vertices);
}

void build_triangles(vector<Triangle>& triangles, vector<vec3>& vertex_indices, vector<vec3>& temp_vertices){

    // For each vertex
    for (int i = 0; i < vertex_indices.size(); i++){

        // Get all the vertices
        vec4 v1 = vec4(temp_vertices[(int)vertex_indices[i].x - 1],1.f);
        vec4 v2 = vec4(temp_vertices[(int)vertex_indices[i].y - 1],1.f);
        vec4 v3 = vec4(temp_vertices[(int)vertex_indices[i].z - 1],1.f);

        // Construct a Triangle with white material (TODO: Implement materials properly)
        Triangle triangle = Triangle(v1, v2, v3, Material(vec3(0.75f, 0.75f, 0.75f)));

        // Append to list of triangles
        triangles.push_back(triangle);
    }

    // TODO: convert all triangles to be in a space between [-1,1]
}