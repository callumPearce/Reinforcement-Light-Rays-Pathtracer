#include "cornell_box_scene.h"

void get_cornell_shapes(std::vector<Triangle>& triangles) {
    
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

    // Triangles now take a material as an argument rather than a colour
    // Floor:
    Triangle flrTri1 = Triangle(C, B, A, green);
    triangles.push_back(flrTri1);

    Triangle flrTri2 = Triangle(C, D, B, green);
    triangles.push_back(flrTri2);

    // Left wall
    Triangle lftWall1 = Triangle(A, E, C, white);
    triangles.push_back(lftWall1);

    Triangle lftWall2 = Triangle(C, E, G, white);
    triangles.push_back(lftWall2);

    // Right wall
    Triangle rghtWall1 = Triangle(F, B, D, white);
    triangles.push_back(rghtWall1);

    Triangle rghtWall2 = Triangle(H, F, D, white);
    triangles.push_back(rghtWall2);

    // Ceiling
    Triangle clng1 = Triangle(E, F, G, cyan);
    triangles.push_back(clng1);

    Triangle clng2 = Triangle(F, H, G, cyan);
    triangles.push_back(clng2);

    // Back wall
    Triangle bckWall1 = Triangle(G, D, C, yellow);
    triangles.push_back(bckWall1);

    Triangle bckWall2 = Triangle(G, H, D, yellow);
    triangles.push_back(bckWall2);

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
    triangles.push_back(Triangle(E,B,A,blue));
    triangles.push_back(Triangle(E,F,B,blue));

    // Front
    triangles.push_back(Triangle(F,D,B,blue));
    triangles.push_back(Triangle(F,H,D,blue));

    // BACK
    triangles.push_back(Triangle(H,C,D,blue));
    triangles.push_back(Triangle(H,G,C,blue));

    // LEFT
    triangles.push_back(Triangle(G,E,C,blue));
    triangles.push_back(Triangle(E,A,C,blue));

    // TOP
    triangles.push_back(Triangle(G,F,E,blue));
    triangles.push_back(Triangle(G,H,F,blue));

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
   
    triangles.push_back(Triangle(E,B,A,red));
    triangles.push_back(Triangle(E,F,B,red));

    // Front
    triangles.push_back(Triangle(F,D,B,red));
    triangles.push_back(Triangle(F,H,D,red));

    // BACK
    triangles.push_back(Triangle(H,C,D,red));
    triangles.push_back(Triangle(H,G,C,red));

    // LEFT
    triangles.push_back(Triangle(G,E,C,red));
    triangles.push_back(Triangle(E,A,C,red));

    // TOP
    triangles.push_back(Triangle(G,F,E,red));
    triangles.push_back(Triangle(G,H,F,red));


    for (size_t i = 0 ; i < triangles.size() ; ++i) {
        triangles[i].setV0(triangles[i].getV0() * (2 / l));
        triangles[i].setV1(triangles[i].getV1() * (2 / l));
        triangles[i].setV2(triangles[i].getV2() * (2 / l));

        triangles[i].setV0(triangles[i].getV0() - vec4(1,1,1,1));
        triangles[i].setV1(triangles[i].getV1() - vec4(1,1,1,1));
        triangles[i].setV2(triangles[i].getV2() - vec4(1,1,1,1));

        vec4 newV0 = triangles[i].getV0();
        newV0.x *= -1;
        newV0.y *= -1;
        newV0.w = 1.0;
        triangles[i].setV0(newV0);

        vec4 newV1 = triangles[i].getV1();
        newV1.x *= -1;
        newV1.y *= -1;
        newV1.w = 1.0;
        triangles[i].setV1(newV1);

        vec4 newV2 = triangles[i].getV2();
        newV2.x *= -1;
        newV2.y *= -1;
        newV2.w = 1.0;
        triangles[i].setV2(newV2);

        triangles[i].compute_and_set_normal();
    }
}