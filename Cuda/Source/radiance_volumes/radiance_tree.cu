#include "radiance_tree.cuh"
#include <algorithm>
#include "printing.h"

__host__
RadianceTree::RadianceTree(){
    this->dimension = X_DIM;
    this->median = 0.f;
}

__host__
RadianceTree::RadianceTree(std::vector<RadianceVolume*>& radiance_volumes, Dimension dimension){
    
    // Set the dimension of the tree
    this->dimension = dimension;

    // Cases based on the number of Radiance Volumes passed in to tree
    int volumes = radiance_volumes.size();
    switch (volumes)
    {
        // No radiance volumes, so a Null tree
        case 0:
            break;
        
        // One radiance volume, so the median is that radiance volume (a leaf node)
        case 1:
            this->median = (float)radiance_volumes[0]->position[dimension];
            this->radiance_volume = new RadianceVolume();
            *(this->radiance_volume) = *(radiance_volumes[0]);
            break;
    
        // Recursive case, more than 1 radiance volume
        default:
            // Sort the radiance volumes on the current dimension
            sort_radiance_volumes_on_dimension(radiance_volumes);
            
            // Find the median to split tree on (gives relatively well balanced tree in practice)
            int median_index = 0;
            if (volumes % 2 == 0){
                median_index = int(volumes/2) - 1;
                this->median = (float)((float)radiance_volumes[median_index]->position[dimension] + (float)radiance_volumes[median_index+1]->position[dimension])/2;
            } else{
                median_index = (int)floor(volumes/2);
                this->median = (float)radiance_volumes[median_index]->position[dimension];
            }

            // Recursively build the tree downwards given the calculated median
            std::vector<RadianceVolume*> left;
            std::vector<RadianceVolume*> right;
            for (int i = 0; i < volumes; i++){
                if (i <= median_index){
                    left.push_back(radiance_volumes[i]);
                } else{
                    right.push_back(radiance_volumes[i]);
                }
            }
            Dimension next_dim = get_next_dimension(dimension);
            this->left_tree = new RadianceTree(left, next_dim);
            this->right_tree = new RadianceTree(right, next_dim);
            break;
    }
}

// Free memory within the tree in a postorder traversal fashion
__host__
RadianceTree::~RadianceTree(){
    // Has a radiance volume, we are at a leaf node
    if (this->radiance_volume != NULL){
        delete this;
    }
    // Recursive: No radiance volume
    else{
        this->left_tree->~RadianceTree();
        this->right_tree->~RadianceTree();
        delete this;
    }
}

// Get the next dimension given the current dimension
__host__ __device__
Dimension RadianceTree::get_next_dimension(Dimension dimension){
    switch (dimension)
    {
        case X_DIM:
            return Y_DIM;
            break;
    
        case Y_DIM:
            return Z_DIM;
            break;
        
        case Z_DIM:
            return X_DIM;
            break;
    }
}

// Compare radiance volumes left and right based on the current dimensions of the tree
__host__
bool RadianceTree::sort_radiance_volumes_on_dimension(std::vector<RadianceVolume*>& radiance_volumes){
    switch (this->dimension)
    {
        case X_DIM:
            std::sort(radiance_volumes.begin(), radiance_volumes.end(), sort_on_x);
            break;
    
        case Y_DIM:
            std::sort(radiance_volumes.begin(), radiance_volumes.end(), sort_on_y);
            break;
        
        case Z_DIM:
            std::sort(radiance_volumes.begin(), radiance_volumes.end(), sort_on_z);
            break;
    }
}

__host__
bool RadianceTree::sort_on_x(RadianceVolume* left, RadianceVolume* right){
    return left->position[0] < right->position[0];
}

__host__
bool RadianceTree::sort_on_y(RadianceVolume* left, RadianceVolume* right){
    return left->position[1] < right->position[1];
}

__host__
bool RadianceTree::sort_on_z(RadianceVolume* left, RadianceVolume* right){
    return left->position[2] < right->position[2];
}


// Converts the radiance tree into an array and returns a pointer to it
__host__
void RadianceTree::convert_to_array(int& radiance_array_size, std::vector<RadianceTreeElement>& radiance_array_v){

    // Preform a traversal of the tree to and adds to the radiance_array_v
    // Add the first element to the tree
    RadianceTreeElement rte = {this->dimension, false, -1, -1, this->median};
    radiance_array_v.push_back(rte);
    this->traverse_and_insert(radiance_array_v, 0);
    radiance_array_size = radiance_array_v.size();
}

// Traverses the treerecusively adding element to vector in traversal order
__host__
void RadianceTree::traverse_and_insert(std::vector<RadianceTreeElement>& radiance_array_v, int parent_idx){

    // Get the current radiance tree element index
    int last_idx = radiance_array_v.size() - 1;

    // Leaf node
    if (this->radiance_volume){
        RadianceTreeElement rtc;

        rtc.dimension = this->dimension;
        rtc.leaf = true;
        rtc.left_idx = -1;
        rtc.right_idx = -1;
        rtc.data = this->radiance_volume->index;

        radiance_array_v[ parent_idx ] = rtc;
    
    }
    // Add the left and right elements to the vector and  recurse
    else{
        radiance_array_v[ parent_idx ].left_idx = last_idx + 1;
        radiance_array_v[ parent_idx ].right_idx = last_idx + 2;
        Dimension dim = get_next_dimension(this->dimension);

        RadianceTreeElement rtc_l;
        rtc_l.dimension = dim;
        rtc_l.leaf = false;
        rtc_l.left_idx = -1;
        rtc_l.right_idx = -1;
        rtc_l.data = this->left_tree->median;

        RadianceTreeElement rtc_r;
        rtc_r.dimension = dim;
        rtc_r.leaf = false;
        rtc_r.left_idx = -1;
        rtc_r.right_idx = -1;
        rtc_r.data = this->right_tree->median;

        radiance_array_v.push_back(rtc_l);
        radiance_array_v.push_back(rtc_r);

        this->left_tree->traverse_and_insert(radiance_array_v, last_idx + 1);
        this->right_tree->traverse_and_insert(radiance_array_v, last_idx + 2);
    }
    // // Only left child, recurse
    // else if (this->left_tree){
    //     radiance_array_v[ parent_idx ].left_idx = last_idx + 1;
    //     Dimension dim = get_next_dimension(this->dimension);
    //     RadianceTreeElement rtc_l = {dim, false, -1, -1, this->left_tree->median};
    //     radiance_array_v.push_back(rtc_l);
    //     this->left_tree->traverse_and_insert(radiance_array_v, last_idx + 1);
    // }
    // // Only right child, recurse
    // else if (this->right_tree){
    //     radiance_array_v[ parent_idx ].right_idx = last_idx + 1;
    //     Dimension dim = get_next_dimension(this->dimension);
    //     RadianceTreeElement rtc_r = {dim, false, -1, -1, this->right_tree->median};
    //     radiance_array_v.push_back(rtc_r);
    //     this->right_tree->traverse_and_insert(radiance_array_v, last_idx + 1);
    // }
}

__host__
int RadianceTree::count_tree_elements(int count){
    if (this->radiance_volume){
        // printf("%d\n",this->radiance_volume->index);
        return count + 1;
    }
    else{
        count = 0;
        count += this->left_tree->count_tree_elements(count);
        count += this->right_tree->count_tree_elements(count);
        return count;
    }
}

__host__
int RadianceTree::count_array_elements(std::vector<RadianceTreeElement>& radiance_array_v){
    int count = test_array_traversal(radiance_array_v, 0);
    // printf("%d\n", count);
    return count;
}

__host__
int RadianceTree::test_array_traversal(std::vector<RadianceTreeElement>& radiance_array_v, int index){

    if (radiance_array_v[index].leaf){
        // printf("%d\n", int(radiance_array_v[index].data));
        return 1;
    }
    else{
        int count = 0;
        // if (radiance_array_v[index].left_idx > -1){
            count +=  test_array_traversal(radiance_array_v, radiance_array_v[index].left_idx);
        // }
        // if (radiance_array_v[index].right_idx > -1){
            count +=  test_array_traversal(radiance_array_v, radiance_array_v[index].right_idx);

            // printf("%d\n", count);
        // }
        return count;
    }
}