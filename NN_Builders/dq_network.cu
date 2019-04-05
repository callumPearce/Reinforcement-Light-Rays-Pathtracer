#include "dq_network.cuh"

// Constructor
DQNetwork::DQNetwork(){
}

// Initialize the networks 
void DQNetwork::initialize(dynet::ParameterCollection& model, int actions_count){
    // Create the networks layers
    // Input: 3x1
    // FC1: 30x1
    // FC2: 100x1
    // Output: action_countx1
    FCLayer FC1 = FCLayer(/*Activation*/ RELU, /*Input dim (rows)*/ 3, /*Output dim (rows)*/ 50, /*Dropout*/ 0.f);
    FCLayer FC2 = FCLayer(/*Activation*/ RELU, /*Input dim (rows)*/ 50, /*Output dim (rows)*/ 200, /*Dropout*/ 0.f);
    FCLayer OUT_FC = FCLayer(/*Activation*/ RELU, /*Input dim (rows)*/ 200, /*Output dim (rows)*/ actions_count, /*Dropout*/ 0.f);

    // Add the parameters of each layer to the vector of maintained params
    FC1.add_params(model, this->params);
    FC2.add_params(model, this->params);
    OUT_FC.add_params(model, this->params);

    // Add layers to the networks vector of layers
    this->layers.push_back(FC1);
    this->layers.push_back(FC2);
    this->layers.push_back(OUT_FC);

    // Set the depth of the network
    this->depth = layers.size();
}

// Performs and inference iteration upon the network with the given input
dynet::Expression DQNetwork::network_inference(dynet::ComputationGraph& graph, dynet::Expression input, bool training){
    
    // Expression for intermediate hidden states
    dynet::Expression h_curr = input;

    // Evaluate each layer
    for (unsigned int i = 0; i < this->depth; i++){
        dynet::Expression h_next = this->layers[i].run_inference(graph, this->params[i], h_curr, training);
        h_curr = h_next;
    }

    // Return the output of the network
    return dynet::softmax(h_curr);
}
