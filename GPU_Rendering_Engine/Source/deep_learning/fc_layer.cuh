#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"

#include <vector>

// The activation function to apply to the layers
enum Activation{
    SIGMOID,
    TANH,
    RELU,
    LINEAR,
    SOFTMAX
};

// Defines a fully connected layer for a neural network
// (a perceptron).
// Note: We do not store params within the layer as this
// would complicate model loading, operations involving params
// must be passed in 
class FCLayer{

    public:

        // Attributes
        unsigned int input_dim;
        unsigned int output_dim;
        Activation activation = LINEAR;
        float dropout = 0.f;

        // Constructor
        FCLayer(Activation a, unsigned int input_dim, unsigned int output_dim, float dropout);

        // Applies the activation function for this layer to the expression 
        dynet::Expression activate(dynet::Expression infer_in);

        // Append parameters to the ParameterCollection passed in
        void add_params(dynet::ParameterCollection& model, std::vector<std::vector<dynet::Parameter>>& params);

        // Performs inference upon the current layer (evaluate layer on inputs)
        // and returns the resulting expression
        // params: Contains weights and biases just for the current layer
        dynet::Expression run_inference(dynet::ComputationGraph& graph, std::vector<dynet::Parameter>& params, dynet::Expression h_infer, bool training);      
};

#endif