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
#define MAX_DIST 0.01f
#define INITIAL_RADIANCE (1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION))*700.f
#define RADIANCE_THRESHOLD (1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION))*400.f

#endif