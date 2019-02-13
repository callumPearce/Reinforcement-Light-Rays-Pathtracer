#include "monte_carlo_test_scene.h"

void get_monte_carlo_shapes(std::vector<Surface>& Surfaces) {
    
    // Materials
    Material blue = Material(vec3(0.15f, 0.15f, 0.75f));
    Material white = Material(vec3(0.75f, 0.75f, 0.75f));
    Material red = Material(vec3(0.75f, 0.15f, 0.15f ));
    Material green = Material(vec3(0.15f, 0.75f, 0.15f));
    Material yellow = Material(vec3(0.75f, 0.75f, 0.15f));
    Material cyan = Material(vec3(0.15f, 0.75f, 0.75f));

    // ---------------------------------------------------------------------------
    // Room

    float l = 555;			// Length of Cornell Box side.

    vec4 A(l,0,0,1);
    vec4 B(0,0,0,1);
    vec4 C(l,0,l,1);
    vec4 D(0,0,l,1);

    vec4 E(l,l,0,1);
    vec4 F(0,l,0,1);
    vec4 G(l,l,l,1);
    vec4 H(0,l,l,1);

    // Surfaces now take a material as an argument rather than a colour
    // Floor:
    Surface flrTri1 = Surface(C, B, A, white);
    Surfaces.push_back(flrTri1);

    Surface flrTri2 = Surface(C, D, B, white);
    Surfaces.push_back(flrTri2);

    // Left wall
    Surface lftWall1 = Surface(A, E, C, white);
    Surfaces.push_back(lftWall1);

    Surface lftWall2 = Surface(C, E, G, white);
    Surfaces.push_back(lftWall2);

    // Right wall
    Surface rghtWall1 = Surface(F, B, D, white);
    Surfaces.push_back(rghtWall1);

    Surface rghtWall2 = Surface(H, F, D, white);
    Surfaces.push_back(rghtWall2);

    // Ceiling
    Surface clng1 = Surface(E, F, G, white);
    Surfaces.push_back(clng1);

    Surface clng2 = Surface(F, H, G, white);
    Surfaces.push_back(clng2);

    // Back wall
    Surface bckWall1 = Surface(G, D, C, white);
    Surfaces.push_back(bckWall1);

    Surface bckWall2 = Surface(G, H, D, white);
    Surfaces.push_back(bckWall2);

    // ---------------------------------------------------------------------------
    // Short block

    A = vec4(240,0,234,1);  //+120 in z -50 in x
    B = vec4( 80,0,185,1);
    C = vec4(190,0,392,1);
    D = vec4( 32,0,345,1);

    E = vec4(240,165,234,1);
    F = vec4( 80,165,185,1);
    G = vec4(190,165,392,1);
    H = vec4( 32,165,345,1);

    // // Front
    // Surfaces.push_back(Surface(E,B,A,blue));
    // Surfaces.push_back(Surface(E,F,B,blue));

    // // Front
    // Surfaces.push_back(Surface(F,D,B,blue));
    // Surfaces.push_back(Surface(F,H,D,blue));

    // // BACK
    // Surfaces.push_back(Surface(H,C,D,blue));
    // Surfaces.push_back(Surface(H,G,C,blue));

    // // LEFT
    // Surfaces.push_back(Surface(G,E,C,blue));
    // Surfaces.push_back(Surface(E,A,C,blue));

    // // TOP
    // Surfaces.push_back(Surface(G,F,E,blue));
    // Surfaces.push_back(Surface(G,H,F,blue));


    for (size_t i = 0 ; i < Surfaces.size() ; ++i) {
        Surfaces[i].setV0(Surfaces[i].getV0() * (2 / l));
        Surfaces[i].setV1(Surfaces[i].getV1() * (2 / l));
        Surfaces[i].setV2(Surfaces[i].getV2() * (2 / l));

        Surfaces[i].setV0(Surfaces[i].getV0() - vec4(1,1,1,1));
        Surfaces[i].setV1(Surfaces[i].getV1() - vec4(1,1,1,1));
        Surfaces[i].setV2(Surfaces[i].getV2() - vec4(1,1,1,1));

        vec4 newV0 = Surfaces[i].getV0();
        newV0.x *= -1;
        newV0.y *= -1;
        newV0.w = 1.0;
        Surfaces[i].setV0(newV0);

        vec4 newV1 = Surfaces[i].getV1();
        newV1.x *= -1;
        newV1.y *= -1;
        newV1.w = 1.0;
        Surfaces[i].setV1(newV1);

        vec4 newV2 = Surfaces[i].getV2();
        newV2.x *= -1;
        newV2.y *= -1;
        newV2.w = 1.0;
        Surfaces[i].setV2(newV2);

        Surfaces[i].compute_and_set_normal();
    }
}