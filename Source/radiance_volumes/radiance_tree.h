#ifndef RADIANCE_TREE_H
#define RADIANCE_TREE_H

#include <glm/glm.hpp>
#include "radiance_volume.h"
#include "radiance_volume_comparator.h"
#include <algorithm>
#include <queue>

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

enum Dimension{
    X_DIM = 0,
    Y_DIM = 1,
    Z_DIM = 2
};

class RadianceTree{

    private:
        float median;
        Dimension dimension;
        vector<RadianceVolume> radiance_volumes;
        RadianceTree* left_tree;
        RadianceTree* right_tree;      

        // Get the next dimension given the current one
        Dimension get_next_dimension(Dimension dimension);

        // Check if left radiance rv is less then right for current 
        //dimension of radiance_tree
        bool sort_radiance_volumes_on_dimension(vector<RadianceVolume>& radiance_volumes);
        static bool sort_on_x(RadianceVolume left, RadianceVolume right);
        static bool sort_on_y(RadianceVolume left, RadianceVolume right);
        static bool sort_on_z(RadianceVolume left, RadianceVolume right);

        // Fill the priority queue with the closest n radiance volumes within max_dist
        // around position
        void populate_closest_volumes_queue(int n, float max_dist, vec4 position, priority_queue<RadianceVolumeComparator>& sorted_queue);

        // Attempt to insert each Radiance Volume in the current tree into the sorted priority queue
        void radiance_volume_sorted_queue_insert(vec4 position, priority_queue<RadianceVolumeComparator>& sorted_queue, float max_dist, int n);
    
    public:
    
        // Constructors
        RadianceTree();
        RadianceTree(vector<RadianceVolume>& radiance_volumes, Dimension dimension);

        // Get the closest n RadianceVolumes within max_dist from position
        vector<RadianceVolume> find_closest_radiance_volumes(int n, float max_dist, vec4 position);
};

#endif