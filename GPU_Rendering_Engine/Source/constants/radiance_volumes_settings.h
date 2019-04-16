/*
    Constants set for radiance volumes populated within the scene
*/

#ifndef RADIANCE_VOLUMES_SETTINGS_H
#define RADIANCE_VOLUMES_SETTINGS_H

// Building
#define GRID_RESOLUTION 8
#define GRID_RHO (1.f/((float)GRID_RESOLUTION*(float)GRID_RESOLUTION))
#define DIAMETER 0.1f
#define AREA_PER_SAMPLE 0.001f

// Querying
#define MAX_DIST 0.005f
#define INITIAL_RADIANCE (1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION))*100.f
#define RADIANCE_THRESHOLD (1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION))*0.1

#define DISTRIBUTION_THRESHOLD 0.f

// MISC
#define SAVE_RADIANCE_MAP true

#endif