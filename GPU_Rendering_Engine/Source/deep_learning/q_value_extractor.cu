#include "q_value_extractor.cuh"

// Convert the vertices into coordinate system w.r.t position
void convert_vertices(
    std::vector<float>& converted_vertices,
    std::vector<float>& vertices,
    vec3 position
){
    int vertices_count = vertices.size();
    for (int v = 0; v < vertices_count; v+=3){
        converted_vertices[ v ] = vertices[v  ] - position.x;
        converted_vertices[v+1] = vertices[v+1] - position.y;
        converted_vertices[v+2] = vertices[v+2] - position.z;
    }
}

// Write q_values into a file for the given position
void write_q_values_for_position(
    DQNetwork dqn,
    vec3 position, 
    vec3 normal, 
    std::vector<float>& vertices,
    std::string output_fname
){
    
    // Convert the vertices into the coord system centred around position
    std::vector<float> converted_vertices(vertices.size());
    convert_vertices(
        converted_vertices,
        vertices,
        position
    );

    // Get the Q-vals
    dynet::ComputationGraph graph;
    dynet::Dim input_dim({(unsigned int)vertices.size()},1);
    dynet::Expression input = dynet::input(graph, input_dim, converted_vertices);                  
    dynet::Expression prediction = dqn.network_inference(graph, input, false);
    std::vector<float> q_vals = dynet::as_vector( graph.forward(prediction));

    // Normalise the q_values
    float q_sum = 0.f;
    for (int i = 0; i < q_vals.size(); i++){
        q_sum += q_vals[i];
    }
    for (int i = 0; i < q_vals.size(); i++){
        q_vals[i] /= q_sum;
    }

    // Write the data to the output file 
    std::ofstream save_file (output_fname, std::ios::app);
    if (save_file.is_open()){
        // Write the position
        save_file << position.x << " " << position.y << " " << position.z << " ";

        // Write the normal
        save_file << " " << normal.x << " " << normal.y << " " << normal.z;

        // Write each Q values
        for (int n = 0; n < q_vals.size(); n++){
            save_file << " " << q_vals[n];
        }

        save_file << "\n";

        // Close the file
        save_file.close();
    }
    else{
        printf("Unable to save the NN Q values.\n");
    }
}


void save_selected_radiance_volumes_vals_nn(
    std::string data_fpath, 
    std::string network_fname,
    Scene& scene,
    int argc,
    char** argv
){

    // file locations
    std::string read_in = data_fpath + "to_select.txt";
    std::string write_out = data_fpath + "selected_deep.txt";

    // Delete the previous radiance volume data file
    std::remove(write_out.c_str());

    // Read in the file for the location of the closest rvs
    std::vector<vec3> volume_locations;
    std::vector<vec3> volume_normals;
    read_hemisphere_locations_and_normals(
        read_in, 
        volume_locations, 
        volume_normals
    );

    // Initialise the network
    auto dyparams = dynet::extract_dynet_params(argc, argv);
    dynet::initialize(dyparams);
    dynet::ParameterCollection model;
    DQNetwork dqn = DQNetwork();
    dqn.initialize(model, scene.vertices_count, GRID_RESOLUTION*GRID_RESOLUTION);
    dynet::TextFileLoader loader(network_fname);
    loader.populate(model);

    // Get the scenes vertices
    std::vector<float> vertices(scene.vertices_count);
    memcpy(&(vertices[0]), scene.vertices, sizeof(float)*scene.vertices_count);

    // For each location,normal pair, write get the q_value predictions
    // from the pre-trained network
    for (int i = 0; i < volume_locations.size(); i++){
        write_q_values_for_position(
            dqn,
            volume_locations[i], 
            volume_normals[i], 
            vertices,
            write_out
        );
    }
}