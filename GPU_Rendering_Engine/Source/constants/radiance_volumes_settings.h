/*
    Constants set for radiance volumes populated within the scene
*/

#ifndef RADIANCE_VOLUMES_SETTINGS_H
#define RADIANCE_VOLUMES_SETTINGS_H

// Building
#define GRID_RESOLUTION 4
#define DIAMETER 0.1f
#define AREA_PER_SAMPLE 0.0002f

// Querying
#define MAX_DIST 0.0002f
#define INITIAL_RADIANCE (1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION))*100.f
#define RADIANCE_THRESHOLD (1.f/((float)GRID_RESOLUTION * (float)GRID_RESOLUTION))*10.f

// MISC
#define SAVE_RADIANCE_MAP true

#endif