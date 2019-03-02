#include "cornell_box_scene.cuh"

void get_cornell_shapes(std::vector<Surface>& Surfaces, std::vector<AreaLight>& light_planes) {
    
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

    vec4 I(l/4, l, (3*l)/4, 1);
    vec4 J((3*l)/4, l, (3*l)/4, 1);
    vec4 K(l/4, l, l/4, 1);
    vec4 L((3*l)/4, l, l/4, 1);

    // Surfaces now take a material as an argument rather than a colour
    // Floor:
    Surface flrTri1 = Surface(C, B, A, green);
    Surfaces.push_back(flrTri1);

    Surface flrTri2 = Surface(C, D, B, green);
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
    Surface clng1 = Surface(F, H, I, cyan);
    Surfaces.push_back(clng1);

    Surface clng2 = Surface(F, I, K, cyan);
    Surfaces.push_back(clng2);

    Surface clng3 = Surface(F, K, E, cyan);
    Surfaces.push_back(clng3);

    Surface clng4 = Surface(K, L, E, cyan);
    Surfaces.push_back(clng4);

    Surface clng5 = Surface(L, G, E, cyan);
    Surfaces.push_back(clng5);

    Surface clng6 = Surface(L, J, G, cyan);
    Surfaces.push_back(clng6);

    Surface clng7 = Surface(I, G, J, cyan);
    Surfaces.push_back(clng7);

    Surface clng8 = Surface(H, G, I, cyan);
    Surfaces.push_back(clng8);

    vec3 diffuse_p = 3.f * vec3(1, 1, 0.9);
    AreaLight a1 = AreaLight(K, I, J, diffuse_p);
    light_planes.push_back(a1);

    AreaLight a2 = AreaLight(K, J, L, diffuse_p);
    light_planes.push_back(a2);

    // Back wall
    Surface bckWall1 = Surface(G, D, C, yellow);
    Surfaces.push_back(bckWall1);

    Surface bckWall2 = Surface(G, H, D, yellow);
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

    // Front
    Surfaces.push_back(Surface(E,B,A,blue));
    Surfaces.push_back(Surface(E,F,B,blue));

    // Front
    Surfaces.push_back(Surface(F,D,B,blue));
    Surfaces.push_back(Surface(F,H,D,blue));

    // BACK
    Surfaces.push_back(Surface(H,C,D,blue));
    Surfaces.push_back(Surface(H,G,C,blue));

    // LEFT
    Surfaces.push_back(Surface(G,E,C,blue));
    Surfaces.push_back(Surface(E,A,C,blue));

    // TOP
    Surfaces.push_back(Surface(G,F,E,blue));
    Surfaces.push_back(Surface(G,H,F,blue));

    // ---------------------------------------------------------------------------
    // Tall block

    A = vec4(443,0,247,1);
    B = vec4(285,0,296,1);
    C = vec4(492,0,406,1);
    D = vec4(334,0,456,1);

    E = vec4(443,330,247,1);
    F = vec4(285,330,296,1);
    G = vec4(492,330,406,1);
    H = vec4(334,330,456,1);

    // Front
   
    Surfaces.push_back(Surface(E,B,A,red));
    Surfaces.push_back(Surface(E,F,B,red));

    // Front
    Surfaces.push_back(Surface(F,D,B,red));
    Surfaces.push_back(Surface(F,H,D,red));

    // BACK
    Surfaces.push_back(Surface(H,C,D,red));
    Surfaces.push_back(Surface(H,G,C,red));

    // LEFT
    Surfaces.push_back(Surface(G,E,C,red));
    Surfaces.push_back(Surface(E,A,C,red));

    // TOP
    Surfaces.push_back(Surface(G,F,E,red));
    Surfaces.push_back(Surface(G,H,F,red));


    for (size_t i = 0 ; i < Surfaces.size() ; ++i) {
        Surfaces[i].v0 = Surfaces[i].v0 * (2 / l);
        Surfaces[i].v1 = Surfaces[i].v1 * (2 / l);
        Surfaces[i].v2 = Surfaces[i].v2 * (2 / l);

        Surfaces[i].v0 = Surfaces[i].v0 - vec4(1,1,1,1);
        Surfaces[i].v1 = Surfaces[i].v1 - vec4(1,1,1,1);
        Surfaces[i].v2 = Surfaces[i].v2 - vec4(1,1,1,1);

        vec4 newV0 = Surfaces[i].v0;
        newV0.x *= -1;
        newV0.y *= -1;
        newV0.w = 1.0f;
        Surfaces[i].v0 = newV0;

        vec4 newV1 = Surfaces[i].v1;
        newV1.x *= -1;
        newV1.y *= -1;
        newV1.w = 1.0f;
        Surfaces[i].v1 = newV1;

        vec4 newV2 = Surfaces[i].v2;
        newV2.x *= -1;
        newV2.y *= -1;
        newV2.w = 1.0f;
        Surfaces[i].v2 = newV2;

        Surfaces[i].compute_and_set_normal();
    }
    
    for (size_t i = 0 ; i < light_planes.size() ; ++i) {
        light_planes[i].v0 = light_planes[i].v0 * (2 / l);
        light_planes[i].v1 = light_planes[i].v1 * (2 / l);
        light_planes[i].v2 = light_planes[i].v2 * (2 / l);

        light_planes[i].v0 = light_planes[i].v0 - vec4(1,1,1,1);
        light_planes[i].v1 = light_planes[i].v1 - vec4(1,1,1,1);
        light_planes[i].v2 = light_planes[i].v2 - vec4(1,1,1,1);

        vec4 newV0 = light_planes[i].v0;
        newV0.x *= -1;
        newV0.y *= -1;
        newV0.w = 1.0f;
        light_planes[i].v0 = newV0;

        vec4 newV1 = light_planes[i].v1;
        newV1.x *= -1;
        newV1.y *= -1;
        newV1.w = 1.0f;
        light_planes[i].v1 = newV1;

        vec4 newV2 = light_planes[i].v2;
        newV2.x *= -1;
        newV2.y *= -1;
        newV2.w = 1.0f;
        light_planes[i].v2 = newV2;

        light_planes[i].compute_and_set_normal();
    }

}