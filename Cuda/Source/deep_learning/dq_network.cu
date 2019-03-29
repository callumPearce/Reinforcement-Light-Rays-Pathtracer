#include "dq_network.cuh"

// Constructor
DQNetwork::DQNetwork(){
}

// Initialize the networks 
void DQNetwork::initialize(dynet::ParameterCollection& model){
    // Create the networks layers
    // Input: 3x1
    // FC1: 30x1
    // FC2: 100x1
    // Output: 900x1 (30x30 angle values)
    FCLayer FC1 = FCLayer(/*Activation*/ RELU, /*Input dim (rows)*/ 3, /*Output dim (rows)*/ 30, /*Dropout*/ 0.f);
    FCLayer FC2 = FCLayer(/*Activation*/ RELU, /*Input dim (rows)*/ 30, /*Output dim (rows)*/ 90, /*Dropout*/ 0.f);
    FCLayer OUT_FC = FCLayer(/*Activation*/ LINEAR, /*Input dim (rows)*/ 90, /*Output dim (rows)*/ GRID_RESOLUTION*GRID_RESOLUTION, /*Dropout*/ 0.f);

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
    return h_curr;
}

// Calculate the networks loss using Q-Learning update rule
dynet::Expression DQNetwork::calculate_loss(
        dynet::ComputationGraph& graph, /* Dynet graph */
        dynet::Expression state_action_q, /* Q(s,a) where s is the current state, a is the action that has been taken */
        dynet::Expression next_state_max_q, /* max_a Q(S_t+1, a) where S_t+1 is the next state, a is the highest next Q val */
        float reward, /* R_t+1, reward for taking a in state s*/
        float discount_factor /* \gamma = cos_theta * BRDF */
    ){
    
    // Convert scalar terms to expressions in the graph
    dynet::Expression r_expr = dynet::input(graph, reward);
    dynet::Expression discount = dynet::input(graph, discount_factor);

    // TODO: Act differently if terminal state TD_Target = R_t+1
    // TD_Target = R_t+1 + \gamma * stop_grad(Q(S_t+1, A_t+1))
    dynet::Expression td_target = r_expr + (discount_factor * next_state_max_q);
    
    // loss = (TD_Target - Q(s,a))^2
    dynet::Expression loss = dynet::pow((td_target - state_action_q), dynet::input(graph, 2.f));    
    return loss;
}