/*
    Constants used for rendering an image with SDL
*/

#ifndef IMAGE_SETTINGS_H
#define IMAGE_SETTINGS_H

#define FULLSCREEN_MODE false
#define SCREEN_WIDTH 512
#define SCREEN_HEIGHT 512
#define FOCAL_LENGTH SCREEN_HEIGHT
#define EPS 0.00001f
#define RHO (1.f / (2.f * 3.1415926535f))
#define ETA 0.3

#define PATH_TRACING_METHOD 0 //0 = default, 1 = reinforcement, 2 = voronoi, 3 = deep learning

#endif