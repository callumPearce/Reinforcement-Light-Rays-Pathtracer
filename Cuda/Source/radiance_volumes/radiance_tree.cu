// #include "radiance_tree.cuh"
// #include <algorithm>
// #include "printing.h"

// RadianceTree::RadianceTree(){
//     this->dimension = X_DIM;
//     this->median = 0.f;
// }

// RadianceTree::RadianceTree(std::vector<RadianceVolume*>& radiance_volumes, Dimension dimension){
    
//     // Set the dimension of the tree
//     this->dimension = dimension;

//     // Cases based on the number of Radiance Volumes passed in to tree
//     int volumes = radiance_volumes.size();
//     switch (volumes)
//     {
//         // No radiance volumes, so a Null tree
//         case 0:
//             break;
        
//         // One radiance volume, so the median is that radiance volume (a leaf node)
//         case 1:
//             this->median = (float)radiance_volumes[0]->position[dimension];
//             this->radiance_volume = new RadianceVolume();
//             *(this->radiance_volume) = *(radiance_volumes[0]);
//             break;
    
//         // Recursive case, more than 1 radiance volume
//         default:
//             // Sort the radiance volumes on the current dimension
//             sort_radiance_volumes_on_dimension(radiance_volumes);
            
//             // Find the median to split tree on (gives relatively well balanced tree in practice)
//             int median_index = 0;
//             if (volumes % 2 == 0){
//                 median_index = int(volumes/2) - 1;
//                 this->median = (float)((float)radiance_volumes[median_index]->position[dimension] + (float)radiance_volumes[median_index+1]->position[dimension])/2;
//             } else{
//                 median_index = (int)floor(volumes/2);
//                 this->median = (float)radiance_volumes[median_index]->position[dimension];
//             }

//             // Recursively build the tree downwards given the calculated median
//             std::vector<RadianceVolume*> left;
//             std::vector<RadianceVolume*> right;
//             for (int i = 0; i < volumes; i++){
//                 if (i <= median_index){
//                     left.push_back(radiance_volumes[i]);
//                 } else{
//                     right.push_back(radiance_volumes[i]);
//                 }
//             }
//             Dimension next_dim = get_next_dimension(dimension);
//             this->left_tree = new RadianceTree(left, next_dim);
//             this->right_tree = new RadianceTree(right, next_dim);
//             break;
//     }
// }

// // Free memory within the tree in a postorder traversal fashion
// RadianceTree::~RadianceTree(){
//     // Has a radiance volume, we are at a leaf node
//     if (this->radiance_volume != NULL){
//         delete this;
//     }
//     // Recursive: No radiance volume
//     else{
//         this->left_tree->~RadianceTree();
//         this->right_tree->~RadianceTree();
//         delete this;
//     }
// }

// // Get the next dimension given the current dimension
// Dimension RadianceTree::get_next_dimension(Dimension dimension){
//     switch (dimension)
//     {
//         case X_DIM:
//             return Y_DIM;
//             break;
    
//         case Y_DIM:
//             return Z_DIM;
//             break;
        
//         case Z_DIM:
//             return X_DIM;
//             break;
//     }
// }

// // Compare radiance volumes left and right based on the current dimensions of the tree
// bool RadianceTree::sort_radiance_volumes_on_dimension(std::vector<RadianceVolume*>& radiance_volumes){
//     switch (this->dimension)
//     {
//         case X_DIM:
//             std::sort(radiance_volumes.begin(), radiance_volumes.end(), sort_on_x);
//             break;
    
//         case Y_DIM:
//             std::sort(radiance_volumes.begin(), radiance_volumes.end(), sort_on_y);
//             break;
        
//         case Z_DIM:
//             std::sort(radiance_volumes.begin(), radiance_volumes.end(), sort_on_z);
//             break;
//     }
// }

// bool RadianceTree::sort_on_x(RadianceVolume* left, RadianceVolume* right){
//     return left->position[0] < right->position[0];
// }

// bool RadianceTree::sort_on_y(RadianceVolume* left, RadianceVolume* right){
//     return left->position[1] < right->position[1];
// }

// bool RadianceTree::sort_on_z(RadianceVolume* left, RadianceVolume* right){
//     return left->position[2] < right->position[2];
// }

// // Get the closest n RadianceVolumes within max_dist from position and having the same normal
// // std::vector<RadianceVolume*> RadianceTree::find_closest_radiance_volumes(int n, float max_dist, vec4 position, vec4 normal){
// //     // Create the priority queue and populate it with the closest radiance volumes
// //     std::priority_queue<RadianceVolumeComparator> sorted_queue;
// //     populate_closest_volumes_queue(n, max_dist, position, normal, sorted_queue);
// //     // Add all radiance volumes in the sorted queue to the list of nearest_volumes
// //     std::vector<RadianceVolume*> nearest_volumes;
// //     while (!sorted_queue.empty()){
// //         RadianceVolumeComparator rvc = sorted_queue.top();
// //         nearest_volumes.push_back(rvc.get_radiance_volume());
// //         sorted_queue.pop();
// //     }
// //     return nearest_volumes;
// // }

// // // Fill the priority queue with the closest n radiance volumes within max_dist
// // // around position
// // void RadianceTree::populate_closest_volumes_queue(int n, float max_dist, vec4 position, vec4 normal, std::priority_queue<RadianceVolumeComparator>& sorted_queue){

// //     float delta = position[this->dimension] - this->median;
// //     int volumes = this->radiance_volumes.size();
// //     // Base Case: There exist radiance volumes on this branch, try to add them in
// //     if (volumes > 0){
// //         radiance_volume_sorted_queue_insert(position, normal, sorted_queue, max_dist, n);
// //     }
// //     // Recursive Case:: No radiance volumes so we recurse down the tree
// //     else{
// //         if (delta < 0){
// //             // Left branch
// //             (this->left_tree)->populate_closest_volumes_queue(n, max_dist, position, normal, sorted_queue);
// //             // Check right branch if it is within the range of max_dist from the point.
// //             // As we may actually be closer to some radiance volumes on the right then
// //             // the ones added on the left still.
// //             if (std::fabs(delta) < std::fabs(max_dist)){
// //                 (this->right_tree)->populate_closest_volumes_queue(n, max_dist, position, normal, sorted_queue);
// //                 if (sorted_queue.size() == n) return;
// //             }
// //         } else{
// //             // Right branch
// //             (this->right_tree)->populate_closest_volumes_queue(n, max_dist, position, normal, sorted_queue);
// //             // Check left branch if it is within the range of max_dist from the point
// //             if (std::fabs(delta) < std::fabs(max_dist)){
// //                 (this->left_tree)->populate_closest_volumes_queue(n, max_dist, position, normal, sorted_queue);
// //                 if (sorted_queue.size() == n) return;
// //             }
// //         }
// //     }
// // }

// // // For the current list of radiance volumes, attempt to add them into the priority queue
// // void RadianceTree::radiance_volume_sorted_queue_insert(vec4 position, vec4 normal, std::priority_queue<RadianceVolumeComparator>& sorted_queue, float max_dist, int n){
// //     int volumes = this->radiance_volumes.size();
// //     for (int i = 0; i < volumes; i++){
// //         float dist = glm::distance(position, this->radiance_volumes[i]->get_position());
// //         RadianceVolumeComparator rvc = RadianceVolumeComparator(this->radiance_volumes[i], dist);
// //         vec3 rv_normal = this->radiance_volumes[i]->get_normal();
// //         // Ensure that the radiance volume is on the same surface
// //         if (rv_normal != vec3(normal)){
// //             continue;
// //         }
// //         if (sorted_queue.size() < n){
// //             sorted_queue.push(rvc);
// //         } else if (rvc < sorted_queue.top()){
// //             sorted_queue.pop();
// //             sorted_queue.push(rvc);
// //         }
// //     }
// // }

// /* GPU ready find closest radiance volume (Singular) */
// // Get the closest radiance volume
// RadianceVolume* RadianceTree::find_closest_radiance_volume(float max_dist, vec4 position, vec4 normal){
//     RadianceVolume temp_rv = RadianceVolume();
//     RadianceVolumeComparator initial_rvc = RadianceVolumeComparator(&temp_rv, 999999.f);
//     RadianceVolumeComparator closest_rvc = find_closest_radiance_volume_comparator(max_dist, position, normal, initial_rvc);
//     if (closest_rvc.radiance_volume->initialised){
//         return closest_rvc.radiance_volume;
//     }
//     else {
//         return NULL;
//     }
// }

// // Get the closest radiance volume comparator recursively
// RadianceVolumeComparator RadianceTree::find_closest_radiance_volume_comparator(float max_dist, vec4 position, vec4 normal, RadianceVolumeComparator current_closest){
//     float delta = position[this->dimension] - this->median;
//     // Base Case: There exist radiance volumes on this branch, try to add them in
//     if (this->radiance_volume != NULL){
//         float dist = glm::distance(position, this->radiance_volume->position);
//         RadianceVolumeComparator rvc = RadianceVolumeComparator(this->radiance_volume, dist);
//         vec3 rv_normal = this->radiance_volume->normal;
//         // Ensure that the radiance volume is on the same surface
//         if (rv_normal == vec3(normal) && rvc < current_closest){
//             current_closest = rvc;
//         }
//     }
//     // Recursive Case:: No radiance volume, so we recurse down the tree
//     else{
//         if (delta < 0){
//             // Left branch
//             current_closest = (this->left_tree)->find_closest_radiance_volume_comparator(max_dist, position, normal, current_closest);
//             // Check right branch if it is within the range of max_dist from the point.
//             // As we may actually be closer to some radiance volumes on the right then
//             // the ones added on the left still.
//             if (std::fabs(delta) < std::fabs(max_dist)){
//                 current_closest = (this->right_tree)->find_closest_radiance_volume_comparator(max_dist, position, normal, current_closest);
//             }
//         } else{
//             // Right branch
//             current_closest = (this->right_tree)->find_closest_radiance_volume_comparator(max_dist, position, normal, current_closest);
//             // Check left branch if it is within the range of max_dist from the point
//             if (std::fabs(delta) < std::fabs(max_dist)){
//                 current_closest = (this->left_tree)->find_closest_radiance_volume_comparator(max_dist, position, normal, current_closest);
//             }
//         }
//     }
//     return current_closest;
// }