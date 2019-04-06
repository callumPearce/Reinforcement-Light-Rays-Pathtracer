#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <unistd.h>
#include "settings.cuh"
#include "dq_network.cuh"

inline bool file_exists (const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

// Read the scene data file and populate the list of scene data
void load_scene_data(std::vector<float>& scene_data){

    // Open the file and read line by line
    std::string line;
    std::ifstream save_file ("../Radiance_Map_Data/nn_scene_data.txt");
    if (save_file.is_open()){

        int line_idx = 0;

        while ( std::getline (save_file, line)){

            // For each space
            size_t pos = 0;
            std::string token;
            while ((pos = line.find(" ")) != std::string::npos){

                // Add the data to the list (Note first 3 numbers are the position)
                token = line.substr(0, pos);
                scene_data.push_back(std::stof(token));

                // Delete the part that we have read
                line.erase(0, pos + 1);
            }

            // Add the final float in
            scene_data.push_back(std::stof(line));
        }
    }
    else{
        printf("Scene Data file could not be opened.\n");
    }
}

// Read the radiance_map data file and populate a vector of vectors
void load_radiance_map_data(std::vector<std::vector<float>>& radiance_map_data, int& action_count){

    // Open the file and read line by line
    std::string line;
    std::ifstream save_file ("../Radiance_Map_Data/radiance_map_data.txt");
    if (save_file.is_open()){

        int line_idx = 0;

        while ( std::getline (save_file, line)){

            if (line_idx == 0){
                action_count = std::stoi(line);
            }
            else{
                // vector of data for the current line read in
                std::vector<float> data_line;
                
                // For each space
                size_t pos = 0;
                std::string token;
                while ((pos = line.find(" ")) != std::string::npos){

                    // Add the data to the list (Note first 3 numbers are the position)
                    token = line.substr(0, pos);
                    data_line.push_back(std::stof(token));

                    // Delete the part that we have read
                    line.erase(0, pos + 1);
                }

                // Add the final float in
                data_line.push_back(std::stof(line));

                // Add data line into the vector
                radiance_map_data.push_back(data_line);
            }
            line_idx++;
        }
    }
    else{
        printf("Radiance Map Data file could not be opened.\n");
    }
}

int main (int argc, char** argv) {

    //////////////////////////////////////////////////////////////
    /*         Read in and shuffle the radiance map data        */
    //////////////////////////////////////////////////////////////
    std::vector<std::vector<float>> radiance_map_data;
    int action_count;
    load_radiance_map_data(radiance_map_data, action_count);
    std::random_shuffle(radiance_map_data.begin(), radiance_map_data.end());
    std::cout << "Read " << radiance_map_data.size() << " lines of radiance_map data." << std::endl;
    std::cout << "Action count: " << action_count << std::endl;

    //////////////////////////////////////////////////////////////
    /*           Read in the scene data                         */
    //////////////////////////////////////////////////////////////
    std::vector<float> scene_data;
    load_scene_data(scene_data);
    std::cout << "Read " << scene_data.size() << " line of scene data." << std::endl;

    //////////////////////////////////////////////////////////////
    /*           Select test and training data                  */
    //////////////////////////////////////////////////////////////
    std::vector<std::vector<float>> training_data;
    std::vector<std::vector<float>> test_data;
    for(int i = 0; i < radiance_map_data.size(); i++){
        // 3 for position, plus the number of actions on each line
        std::vector<float> data_line(3 + action_count);
        std::copy( radiance_map_data[i].begin(), radiance_map_data[i].end(), data_line.begin());

        double rv = ((double) rand() / (RAND_MAX));

        // Training
        if( rv < 0.8  ){
            training_data.push_back(data_line);
        }
        // Test
        else{
            test_data.push_back(data_line);
        }
    }

    std::cout << "Training data set size: " << training_data.size() << std::endl;
    std::cout << "Test data set size: " << test_data.size() << std::endl;

    //////////////////////////////////////////////////////////////
    /*                Initialise the DNN                        */
    //////////////////////////////////////////////////////////////
    auto dyparams = dynet::extract_dynet_params(argc, argv);
    dynet::initialize(dyparams);
    dynet::ParameterCollection model;
    dynet::AdamTrainer trainer(model);
    DQNetwork dnn = DQNetwork();
    dnn.initialize(model, 3+scene_data.size(), action_count);

    //////////////////////////////////////////////////////////////
    /*             Load in the Parameter Values                 */
    //////////////////////////////////////////////////////////////
    std::string fname = "../Radiance_Map_Data/radiance_map_model.model";
    if (LOAD_MODEL && file_exists(fname)){
        dynet::TextFileLoader loader(fname);
        loader.populate(model);
    }

    //////////////////////////////////////////////////////////////
    /*              Train and test the DNN                      */
    //////////////////////////////////////////////////////////////
    unsigned int num_batches = training_data.size() / BATCH_SIZE - 1;

    for (int i = 0; i < EPOCHS; i++){

        /* Train */
        float loss = 0.f;
        for (int b = 0; b < num_batches; b++){

            // Initialise the computation graph
            dynet::ComputationGraph graph;

            // Batch start id and current batch size
            unsigned int sidx = b*BATCH_SIZE;
            unsigned int batch_size = std::min((unsigned int) training_data.size() - sidx, (unsigned int) BATCH_SIZE);

            // Get the input data and targets for the current batch
            std::vector<dynet::Expression> current_targets(batch_size);
            std::vector<dynet::Expression> current_batch(batch_size);
            for( unsigned int k = 0; k < batch_size; k++){

                unsigned int data_idx = sidx+k;

                // Get the inputs
                std::vector<float> batch_targets(training_data[data_idx].begin()+3, training_data[data_idx].end());
                
                // Get the outputs
                std::vector<float> batch_inputs(3 + scene_data.size());
                memcpy(
                    &(batch_inputs[0]),
                    &(training_data[data_idx][0]),
                    sizeof(float) * 3
                );
                memcpy(
                    &(batch_inputs[3]),
                    &(scene_data[0]),
                    sizeof(float) * scene_data.size()
                );

                // std::vector<float> batch_inputs(training_data[data_idx].begin(), training_data[data_idx].begin()+3);

                current_targets[k] = dynet::input(graph, {(unsigned int)action_count}, batch_targets);
                current_batch[k] = dynet::input(graph, {3 + (unsigned int)scene_data.size()}, batch_inputs);
            }
            dynet::Expression targets_batch = reshape(concatenate_cols(current_targets), dynet::Dim({(unsigned int) action_count}, batch_size));
            dynet::Expression input_batch = reshape(concatenate_cols(current_batch), dynet::Dim({3 + (unsigned int)scene_data.size()}, batch_size));

            // Forward pass through the network
            dynet::Expression output_batch = dnn.network_inference(graph, input_batch, true);

            // Define the loss
            dynet::Expression loss_expr = dynet::sum_batches(dynet::squared_distance(targets_batch, output_batch));
            loss += dynet::as_scalar(graph.forward(loss_expr));

            graph.backward(loss_expr);
            trainer.update();
        }

        float error = 0.f;

        /* Test */
        for (int t = 0; t < test_data.size(); t++){

            // Initialise the computational graph
            dynet::ComputationGraph graph;

            // Get the input expression and ground truth
            std::vector<float> targets(test_data[t].begin()+3, test_data[t].end());
            // std::vector<float> inputs(test_data[t].begin(), test_data[t].begin()+4);

            // Get the outputs
            std::vector<float> inputs(3 + scene_data.size());
            memcpy(
                &(inputs[0]),
                &(test_data[t][0]),
                sizeof(float) * 3
            );
            memcpy(
                &(inputs[3]),
                &(scene_data[0]),
                sizeof(float) * scene_data.size()
            );

            dynet::Expression input = dynet::input(graph, {3 + (unsigned int)scene_data.size()}, inputs);
            dynet::Expression ground_truth = dynet::input(graph, {(unsigned int) action_count}, targets);

            // Get the predictions
            dynet::Expression prediciton = dnn.network_inference(graph, input, false);

            // Calulate difference between prediction and ground truth
            dynet::Expression diff = dynet::squared_distance(ground_truth, prediciton);

            error += dynet::as_scalar(graph.forward(diff)); 
        }

        std::cout << "---------------- " << i+1 << " ----------------" << std::endl;
        std::cout << "      Loss: " << loss << std::endl;
        std::cout << "      Error: " << error << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        std::cout << std::endl;
    }

    //////////////////////////////////////////////////////////////
    /*                      Save the Model                      */
    //////////////////////////////////////////////////////////////
    if (SAVE_MODEL){
        dynet::TextFileSaver saver("../Radiance_Map_Data/radiance_map_model.model");
        saver.save(model);
    }

    return 0;
}