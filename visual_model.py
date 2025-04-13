import numpy as np
from model import neuralNetwork
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
parser = argparse.ArgumentParser(description='Hyperparameters for training neural network')
parser.add_argument("--model", type=str, help='Path to model.pkl')

args = parser.parse_args()

if __name__ == '__main__':
    model_path = args.model
    with open(model_path, 'rb') as f:
        model_parameters = pickle.load(f)
    weights_input = model_parameters['input_layer_weights']
    weights_hidden = model_parameters['hidden_layer_weights']
    weights_output = model_parameters['output_layer_weights']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    weights_input_flatten = weights_input.flatten()
    # sns.heatmap(weights_input, annot=True, cmap="YlGnBu", fmt=".2f")
    axes[0].hist(weights_input_flatten, bins=50, color='blue', edgecolor='black')
    axes[0].set_title("Weights distribution of input layer")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(axis='y', linestyle='--')
    # plt.savefig("Input layer weights")

    # plt.figure()
    weights_hidden_flatten = weights_hidden.flatten()
    # sns.heatmap(weights_input, annot=True, cmap="YlGnBu", fmt=".2f")
    axes[1].hist(weights_hidden_flatten, bins=50, color='blue', edgecolor='black')
    axes[1].set_title("Weights distribution of hidden layer")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(axis='y', linestyle='--')
    # axes[1].savefig("Hidden layer weights")

    # plt.figure()
    weights_output_flatten = weights_output.flatten()
    # sns.heatmap(weights_input, annot=True, cmap="YlGnBu", fmt=".2f")
    axes[2].hist(weights_output_flatten, bins=50, color='blue', edgecolor='black')
    axes[2].set_title("Weights distribution of output layer")
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(axis='y', linestyle='--')
    plt.savefig("Layer weights distribution")
