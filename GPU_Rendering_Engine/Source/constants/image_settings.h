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
#define RHO (1.f / (3.1415926535f))
#define EPSILON_START 0.6f
#define EPSILON_MIN 0.05f
#define EPSILON_DECAY 0.01f

#define PATH_TRACING_METHOD 0 //0 = default, 1 = reinforcement, 2 = voronoi, 3 = deep reinforcement learning, 4 = pretrained network inference
#define LOAD_MODEL true
#define SAVE_MODEL true

#endif