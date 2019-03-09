#ifndef RADIANCE_TREE_H
#define RADIANCE_TREE_H

#include <glm/glm.hpp>
#include "radiance_volume.cuh"
#include "radiance_volume_comparator.cuh"

using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;

enum Dimension{
    X_DIM = 0,
    Y_DIM = 1,
    Z_DIM = 2
};

struct RadianceTreeElement{
    Dimension dimension;
    bool leaf;
    int left_idx;
    int right_idx;
    float data;
};

class RadianceTree{

    private:   

        // Check if left radiance rv is less then right for current 
        //dimension of radiance_tree
        __host__
        bool sort_radiance_volumes_on_dimension(std::vector<RadianceVolume*>& radiance_volumes);

        __host__
        static bool sort_on_x(RadianceVolume* left, RadianceVolume* right);

        __host__
        static bool sort_on_y(RadianceVolume* left, RadianceVolume* right);

        __host__
        static bool sort_on_z(RadianceVolume* left, RadianceVolume* right);

        // Recursively traverse the tree adding elements to the supplied vector
        __host__
        void traverse_and_insert(std::vector<RadianceTreeElement>& radiance_array_v, int parent_idx);

        __host__
        static int test_array_traversal(std::vector<RadianceTreeElement>& radiance_array_v, int index);

    public:

        float median = 0.f;
        Dimension dimension;
        RadianceVolume* radiance_volume = NULL;
        RadianceTree* left_tree;
        RadianceTree* right_tree;
        RadianceTreeElement* radiance_array;

        // Constructors
        __device__
        RadianceTree();

        __host__
        RadianceTree(std::vector<RadianceVolume*>& radiance_volumes, Dimension dimension);

        // Destructor
        __host__
        ~RadianceTree();

        // Convert to an array representation
        __host__
        void convert_to_array(int& radiance_array_size, std::vector<RadianceTreeElement>& radiance_array_v);

        // Get the next dimension given the current one
        __host__ __device__
        static Dimension get_next_dimension(Dimension dimension);

        __host__
        static int count_array_elements(std::vector<RadianceTreeElement>& radiance_array_v);

        __host__
        int count_tree_elements(int count);
};

#endif