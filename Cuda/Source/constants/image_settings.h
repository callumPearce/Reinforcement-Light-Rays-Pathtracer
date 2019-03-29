/*
    Constants used for rendering an image with SDL
*/

#ifndef IMAGE_SETTINGS_H
#define IMAGE_SETTINGS_H

#define FULLSCREEN_MODE false
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 128
#define FOCAL_LENGTH SCREEN_HEIGHT
#define EPS 0.00001f
#define RHO (1.f / (2.f * 3.1415926535f))

#define PATH_TRACING_METHOD 3 //0 = default, 1 = reinforcement, 2 = voronoi, 3 = deep learning

#endif