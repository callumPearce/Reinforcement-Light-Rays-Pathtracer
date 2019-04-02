#ifndef SDL_SCREEN_H
#define SDL_SCREEN_H

#include "SDL.h"
#include <iostream>
#include <glm/glm.hpp>
#include <stdint.h>



class SDLScreen{

    private:
      SDL_Window *window;
      SDL_Renderer *renderer;
      SDL_Texture *texture;

    public:
        uint32_t *buffer;
        int height;
        int width;

        SDLScreen( int width, int height, bool fullscreen = false );
        bool NoQuitMessageSDL();
        void PutPixelSDL(int x, int y, glm::vec3 color );
        void SDL_Renderframe();
        void SDL_SaveImage(const char* filename);
        void kill_screen();

};

#endif