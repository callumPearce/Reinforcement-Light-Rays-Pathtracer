#include "sdl_screen.h"

SDLScreen::SDLScreen(int width, int height, bool fullscreen) {
  if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) !=0)
    {
      printf("Could not initialise SDL: %s\n", SDL_GetError());
      exit(1);
    }
  
  this->width = width;
  this->height = height;
  this->buffer = new uint32_t[width*height];
  memset(this->buffer, 0, width*height*sizeof(uint32_t));
  
  uint32_t flags = SDL_WINDOW_OPENGL;
  if(fullscreen)
    {
      flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    }
  this->window = SDL_CreateWindow("Reinforcement Raytracer",
				      SDL_WINDOWPOS_UNDEFINED,
				      SDL_WINDOWPOS_UNDEFINED,
				      width, height,flags);
  if(this->window == 0)
    {
      printf("Could not set video mode: %s\n", SDL_GetError());
      exit(1);
    }

  flags = SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC;
  this->renderer = SDL_CreateRenderer(this->window, -1, flags);
  if(this->renderer == 0)
    {
      printf("Could not create renderer: %s\n", SDL_GetError());
      exit(1);
    }
  SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");
  SDL_RenderSetLogicalSize(this->renderer, width,height);

  this->texture = SDL_CreateTexture(this->renderer,
				 SDL_PIXELFORMAT_ARGB8888,
				 SDL_TEXTUREACCESS_STATIC,
				 this->width,this->height);
  if(this->texture==0)
    {
      printf("Could not allocate texture: %s\n", SDL_GetError());
      exit(1);
    }
}

void SDLScreen::kill_screen(){
  delete[] this->buffer;
  SDL_DestroyTexture(this->texture);
  SDL_DestroyRenderer(this->renderer);
  SDL_DestroyWindow(this->window); 
  SDL_Quit();
}


void SDLScreen::SDL_SaveImage(const char* filename)
{
  SDL_Surface *sshot = SDL_CreateRGBSurface(0, this->width, this->height, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
  SDL_RenderReadPixels(this->renderer, NULL, SDL_PIXELFORMAT_ARGB8888, sshot->pixels, sshot->pitch);
  SDL_SaveBMP(sshot, filename);
  SDL_FreeSurface(sshot);
}

void SDLScreen::SDL_Renderframe()
{
  SDL_UpdateTexture(this->texture, NULL, this->buffer, this->width*sizeof(uint32_t));
  SDL_RenderClear(this->renderer);
  SDL_RenderCopy(this->renderer, this->texture, NULL, NULL);
  SDL_RenderPresent(this->renderer);
}

bool SDLScreen::NoQuitMessageSDL()
{
  SDL_Event e;
  while( SDL_PollEvent(&e) )
    {
      if( e.type == SDL_QUIT )
	{
	  return false;
	}
      if( e.type == SDL_KEYDOWN )
	{
	  if( e.key.keysym.sym == SDLK_ESCAPE)
	    {
	      return false;
	    }
	}
    }
  return true;
}

void SDLScreen::PutPixelSDL(int x, int y, glm::vec3 colour)
{
  if(x<0 || x>=this->width || y<0 || y>=this->height)
    {
      printf("apa\n");
      return;
    }
  uint32_t r = uint32_t( glm::clamp( 255*colour.r, 0.f, 255.f ) );
  uint32_t g = uint32_t( glm::clamp( 255*colour.g, 0.f, 255.f ) );
  uint32_t b = uint32_t( glm::clamp( 255*colour.b, 0.f, 255.f ) );

  this->buffer[y*this->width+x] = (128<<24) + (r<<16) + (g<<8) + b;
}
