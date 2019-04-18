import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def read_data_nn_training_stats(filename, avg_path_lengths, training_loss, zero_contrib_paths):

    # Append the Radiance Map directory on the front of the file
    filename = "../Radiance_Map_Data/" + filename

    # Check if file does not exist
    if not os.path.isfile(filename):
        return

    # Open the file and read its data line by line
    with open(filename) as f:
        for line in f:
            data_line = line.split(" ")
            avg_path_lengths.append(float(data_line[0]))
            training_loss.append(float(data_line[1]))
            zero_contrib_paths.append(int(data_line[2].strip("\n")))
    

def plot_avg_path_lengths(avg_path_lengths):

    # Convert to np array for plotting
    y_data = np.asarray(avg_path_lengths)
    x_data = np.arange(len(y_data))

    # Plot the data
    path_length_fig = plt.figure()
    plt.plot(x_data, y_data)
    plt.xlabel("Epochs")
    plt.ylabel("Average Path Length")
    plt.title("Average Path Length for Neural-Q Path Tracer")


def plot_training_loss(training_loss):

    # Convert to np array for plotting
    y_data = np.asarray(training_loss)
    x_data = np.arange(len(y_data))

    # Plot the data
    traing_loss_fig = plt.figure()
    plt.plot(x_data, y_data)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss for Neural-Q Path Tracer")


def plot_zero_contrib_paths(zero_contrib_paths):

    # Convert to np array for plotting
    y_data = np.asarray(zero_contrib_paths)
    x_data = np.arange(len(y_data))

    # Plot the data
    traing_loss_fig = plt.figure()
    plt.plot(x_data, y_data)
    plt.xlabel("Epochs")
    plt.ylabel("Number of zero contribution light paths")
    plt.title("Number of zero contribution light paths for Neural-Q Path Tracer")


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("No filename name to read the stats from. Terminating.")
    else:

        # Setup storage for training stats
        avg_path_lengths = []
        training_loss = []
        zero_contrib_paths = []

        # Read in the training stats
        read_data_nn_training_stats(sys.argv[1], avg_path_lengths, training_loss, zero_contrib_paths)

        # Plot the avg_path_length
        plot_avg_path_lengths(avg_path_lengths)

        # Plot the training loss
        plot_training_loss(training_loss)

        # Plot the zero contribution paths
        plot_zero_contrib_paths(zero_contrib_paths)

        # Show the Plots
        plt.show()
