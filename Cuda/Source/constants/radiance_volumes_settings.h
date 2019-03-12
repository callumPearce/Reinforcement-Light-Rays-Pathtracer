/*
    Constants set for radiance volumes populated within the scene
*/

#ifndef RADIANCE_VOLUMES_SETTINGS_H
#define RADIANCE_VOLUMES_SETTINGS_H

// Building
#define GRID_RESOLUTION 40
#define DIAMETER 0.1f
#define AREA_PER_SAMPLE 0.0005f

// Querying
#define CLOSEST_QUERY_COUNT 1
#define MAX_DIST 0.0005f
#define RADIANCE_THRESHOLD (1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION))// /6.f

#endif