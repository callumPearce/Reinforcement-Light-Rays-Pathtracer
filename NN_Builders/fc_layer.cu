#include "fc_layer.cuh"

FCLayer::FCLayer(Activation a, unsigned int input_dim, unsigned int output_dim, float dropout){
    // Set variables
    this->activation = a;
    this->input_dim = input_dim;
    this->output_dim = output_dim;
    this->dropout = dropout;
}

// Applies the activation function for this layer to the expression
dynet::Expression FCLayer::activate(dynet::Expression infer_in){
    switch (this->activation) {
        case LINEAR:
          return infer_in;
        case RELU:
          return dynet::rectify(infer_in);
        case SIGMOID:
          return dynet::logistic(infer_in);
        case TANH:
          return dynet::tanh(infer_in);
        case SOFTMAX:
          return dynet::softmax(infer_in);
    }
    return infer_in;
}

// Append parameters to the ParameterCollection passed in
void FCLayer::add_params(dynet::ParameterCollection& model, std::vector<std::vector<dynet::Parameter>>& params){
    // Add weights and biases (Xavier Initialization used by default) to
    // static model 
    dynet::Parameter W = model.add_parameters({this->output_dim, this->input_dim}); /* Output first as affine transform will be W*x*/
    dynet::Parameter b = model.add_parameters({this->output_dim});
    params.push_back({W,b});
}

// Performs inference upon the current layer (evaluate layer on inputs)
// and returns the resulting expression
// params: Contains weights and biases just for the current layer
dynet::Expression FCLayer::run_inference(dynet::ComputationGraph& graph, std::vector<dynet::Parameter>& params, dynet::Expression h_infer, bool training){
    
    // Load the parameters into the computation graph
    dynet::Expression W = dynet::parameter(graph, params[0]);
    dynet::Expression b = dynet::parameter(graph, params[1]);
    
    // Apply affine transformation: b + W * x = output 
    dynet::Expression infer_out = dynet::affine_transform({b, W, h_infer});
    
    // Apply activation function
    dynet::Expression infer_out_a = this->activate(infer_out);

    // Apply dropout if training
    dynet::Expression infer_next;
    if (this->dropout > 0.f){
        if(training){
            // Vector of 1's and 0's where (1 - dropout) are 1's
            dynet::Expression mask = dynet::random_bernoulli(graph, {this->output_dim}, 1 - this->dropout);
            // Multiply by the mask for dropout (componentwise)
            infer_next = dynet::cmult(infer_out_a, mask);
        }
        // Else, multiply by retention to scale
        else{
            infer_next = infer_out_a * (1 - this->dropout);
        } 
    }
    else{
        infer_next = infer_out_a;
    }

    // Return the output expression from the layer
    return infer_next;
}