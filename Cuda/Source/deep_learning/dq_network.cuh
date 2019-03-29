#ifndef DQ_NETWORK_H
#define DQ_NETWORK_H

#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

#include "fc_layer.cuh"
#include "radiance_volumes_settings.h"

class DQNetwork{

    private:

        // DQNs fully connected layers
        std::vector<FCLayer> layers;
        // Parameters for the network (saved here to make saving them much easier)
        std::vector<std::vector<dynet::Parameter>> params;
        // Layer count
        unsigned int depth;

    public:

        // Constructor
        DQNetwork();

        // Initialize the network with its parameters
        void initialize(dynet::ParameterCollection& model);

        // Performs and inference iteration upon the network with the given input
        dynet::Expression network_inference(dynet::ComputationGraph& graph, dynet::Expression input, bool training);

        // Calculate the loss for the given input within the networks
        dynet::Expression calculate_loss(dynet::ComputationGraph& graph, dynet::Expression input, dynet::Expression next_state_max_q);

        // Calculate the networks loss using Q-Learning update rule
        dynet::Expression calculate_loss(
            dynet::ComputationGraph& graph, /* Dynet graph */
            dynet::Expression state_action_q, /* Q(s,a) where s is the current state, a is the action that has been taken */
            dynet::Expression next_state_max_q, /* max_a Q(S_t+1, a) where S_t+1 is the next state, a is the highest next Q val */
            float reward, /* R_t+1, reward for taking a in state s*/
            float discount_factor /* \gamma = cos_theta * BRDF */
        );
};

#endif