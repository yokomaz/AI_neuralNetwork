import numpy as np
from model import neuralNetwork#, neuralNetwork_v2
from dataset import Dataset, DataLoader
from tqdm import tqdm
from utils import one_hot_encode
import argparse
import pickle
from loss_function import validation
import copy

parser = argparse.ArgumentParser(description='Hyperparameters for training neural network')
parser.add_argument("--model", type=str, help='Path to model.pkl')
args = parser.parse_args()

if __name__ == '__main__':
    model_path = args.model

    print(f"Preparing test dataset")
    dataset = Dataset.load("dataset_cifar10")
    test_set = dataset.test
    x_test = test_set['data']
    y_test = one_hot_encode(test_set['label'])
    y_label = test_set['label']

    test_loader = DataLoader(x_test, y_test, batch_size=len(y_label))#batch_size, shuffle=False)
    print(f"Done")

    print(f"Preparing Model...")
    input_dim = x_test.shape[1]
    output_dim = y_test.shape[1]
    with open(model_path, 'rb') as f:
        model_parameters = pickle.load(f)
    hidden_dim = model_parameters["hidden_layer_weights"].shape[1]
    print(f"Hidden dimension is {hidden_dim}")
    model = neuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    print(f"input_layer before {model.input_layer.weights}")
    initial_model_params = copy.deepcopy(model.input_layer.weights)
    model.load(model_path)
    print(f"input_layer after {model.input_layer.weights}")
    loaded_model_params = model.input_layer.weights
    # exit()
    print(f"Done")

    _, test_accuracy = validation(model, val_dataloader=test_loader)
    # loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True, colour='yellow')
    # #label_pred = []
    # for batch_idx, (batch_x, batch_y) in loop:
    #     print(batch_x == x_test)
    #     exit()
    #     loop.set_description(f'Test step: [{batch_idx+1}/{test_loader.__len__()}]')
    #     y_pred = model.forward(batch_x)
        
    #     # print(y_pred[0])
    #     y_pred = np.argmax(y_pred, axis=1)
    #     print(y_pred)
    #     # label_pred.append(label)
    # # label_pred = np.concatenate(label_pred).reshape(-1, 1)
    # # print(label)
    # # exit()
    # y_pred = y_pred.reshape(-1, 1)
    # accuracy = np.mean(y_label == y_pred)#  / len(y_label) #label_pred) / len(y_label)
    print(f"Accuracy over test set is {100 * test_accuracy:.2f}%")