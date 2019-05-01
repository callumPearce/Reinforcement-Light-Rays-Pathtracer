/*
    Constants used for rendering an image with SDL
*/

#ifndef IMAGE_SETTINGS_H
#define IMAGE_SETTINGS_H

#define FULLSCREEN_MODE false
#define SCREEN_WIDTH 720
#define SCREEN_HEIGHT 720
#define FOCAL_LENGTH SCREEN_HEIGHT
#define EPS 0.00001f
#define RHO (1.f / (2.f*3.1415926535f))

#define RENDER_SAVED_RADIANCE_VOLUMES false

#define PATH_TRACING_METHOD 3
//0 = default
//1 = reinforcement: expected SARSA
//2 = voronoi
//3 = deep reinforcement learning
//4 = pretrained network inference
//5 = save nn q_value predictions for a given point (not for rendering)

#endif