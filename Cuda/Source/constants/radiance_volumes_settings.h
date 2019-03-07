/*
    Constants set for radiance volumes populated within the scene
*/

#ifndef RADIANCE_VOLUMES_SETTINGS_H
#define RADIANCE_VOLUMES_SETTINGS_H

// Building
#define GRID_RESOLUTION 30
#define DIAMETER 0.1f
#define AREA_PER_SAMPLE 0.01f

// Querying
#define CLOSEST_QUERY_COUNT 1
#define MAX_DIST 50.f
#define RADIANCE_THRESHOLD (1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION))// /6.f

#endif