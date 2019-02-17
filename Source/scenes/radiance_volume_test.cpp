#include "radiance_volume_test.h"
#include "radiance_volume.h"
#include "radiance_volumes_settings.h"
#include "printing.h"

// Return the traingles for a single radiance sphere in the middle of the room
void get_radiance_volume_shapes(vector<Surface>& surfaces){
    // Create the radiance volume object
    RadianceVolume rv = RadianceVolume(vec4(0.f, 1.f, 0.f, 1.f), vec4(0, -1.f, 0, 1.f));
    // Get its vertices
    vector<vector<vec4>> vertices;
    rv.get_vertices(vertices);
    // Build Surfaces using the vertices
    for (int x = 0; x < GRID_RESOLUTION; x++){
        for (int y = 0; y < GRID_RESOLUTION; y++){
            vec4 v1 = vertices[x][y];
            vec4 v2 = vertices[x+1][y];
            vec4 v3 = vertices[x][y+1];
            vec4 v4 = vertices[x+1][y+1];
            float r1 = ((float) rand() / (RAND_MAX));
            float r2 = ((float) rand() / (RAND_MAX));
            float r3 = ((float) rand() / (RAND_MAX));
            surfaces.push_back(Surface(v1, v3, v2, Material(vec3(r1, r2, r3))));
            surfaces.push_back(Surface(v2, v3, v4, Material(vec3(r1, r2, r3))));
        }
    }
}