import numpy as np
from model import neuralNetwork#, neuralNetwork_v2
from dataset import Dataset, DataLoader
from utils import one_hot_encode, lr_scheduler, cos_lr_scheduler, SGD
from sklearn.model_selection import train_test_split
from loss_function import cross_entropy, validation
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(description='Hyperparameters for training neural network')
parser.add_argument("--learning_rate", type=float, help='Learning rate for training, default=1e-4', default=1e-4)
parser.add_argument("--activate_func", type=str, help='Activation function for neural network [relu, sigmoid], default=relu', default='relu')
parser.add_argument("--hidden_dim", type=int, help='Hidden layer size, default=2304', default=256)
parser.add_argument("--l2_lambda", type=float, help='L2 strength for loss function, default=1e-6', default=1e-3)
parser.add_argument("--batch_size", type=int, help='Batch size for training, default=100', default=256)
parser.add_argument("--nb_epochs", type=int, help='Number of epochs during training, default=50', default=200)

args = parser.parse_args()

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot编码标签
    num_classes = 10
    y_train = np.eye(num_classes)[y_train.flatten()]
    y_test = np.eye(num_classes)[y_test.flatten()]
    
    return x_train, y_train, x_test, y_test

if __name__=="__main__":
    # hyperparameters
    nb_epoch = args.nb_epochs           # 50
    learning_rate = args.learning_rate  # 1e-4
    activation = args.activate_func     # 'sigmoid'
    l2_lambda = args.l2_lambda          # 1e-4
    hidden_dim = args.hidden_dim        # 2304
    batch_size = args.batch_size        # 100

    # with open("training_log.txt", "a") as f:
        # f.write(f"LR: {learning_rate}, Hidden: {hidden_dim}, Best Val Acc: {best_accuracy_val:.2f}%\n")

    print(f"Hyperparameters are: nb_epoch {nb_epoch}, learning_rate {learning_rate}, activation function {activation}, l2_lambda {l2_lambda}, hidden_dim {hidden_dim}, batch_size {batch_size}")

    # prepare dataset and do some preprocessing
    print(f"Preparing dataset")
    dataset = Dataset.load("dataset_cifar10")
    train_set = dataset.train
    test_set = dataset.test
    
    x_train, x_val, y_train, y_val = train_test_split(train_set['data'], train_set['label'], \
                                                      random_state=42, test_size=0.2, shuffle=True, stratify=train_set['label'])
    print(f"Done")
    #unique_labels, counts = np.unique(y_train, return_counts=True)
    #for unique_labels, counts in zip(unique_labels, counts):
    #    print(unique_labels, counts)
    #exit()

    print(f"Convering labels to one-hot-encode")
    y_train = one_hot_encode(y_train)
    y_val = one_hot_encode(y_val)
    test_set['label'] = one_hot_encode(test_set['label'])
    x_test = test_set['data']
    y_test = test_set['label']

    train_loader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
    train_accuracy_loader = DataLoader(x_train, y_train, batch_size=len(y_train))
    val_loader = DataLoader(x_val, y_val, batch_size=len(y_val))#batch_size)
    test_loader = DataLoader(x_test, y_test, batch_size=len(y_test))

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    # print(f"x_train {x_train.shape}, y_train {y_train.shape}, x_val {x_val.shape}, y_val {y_val.shape}, x_test {x_test.shape}, y_test {y_test.shape}")

    # training set
    lr_schechlar = lr_scheduler(learning_rate, gamma=0.95)
    # cos_lr_scheduler = cos_lr_scheduler(lr_start=learning_rate/2, lr_max=learning_rate, lr_min=0, warm_up_epoch=10, max_epoch=nb_epoch)
    Model = neuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, activate=activation)
    Model.save("Initial_model.pkl")
    optim = SGD(Model.parameters, lr=lr_schechlar)
    train_loss = []
    validation_loss = []
    validation_accuracy = []
    # test_accuracy_list = []
    best_accuracy_val = 0
    lr_buffer = []
    train_accuracy_buffer = []

    # start training
    for i in range(nb_epoch):
        lr = lr_schechlar.get_lr()
        # lr = cos_lr_scheduler.get_lr()
        lr_buffer.append(lr)
        average_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, colour='green')
        for batch_idx, (batch_x, batch_y) in loop:
            y_pred = Model.forward(batch_x)
            loss = cross_entropy(y_pred, batch_y, Model, l2_lambda=l2_lambda)
            average_loss += loss
            Model.backward(batch_y, l2_lambda, learning_rate=lr)
            # print(f"before step {Model.input_layer.weights}")
            optim.step()
            optim.zero_grad()
            # print(f"after step {Model.input_layer.weights}")
            # print(id(Model.input_layer.weights))
            # print(id(Model.parameters['input_layer_weights']))
            # exit()

            loop.set_description(f'Train Epoch: [{i+1}/{nb_epoch}]')
            loop.set_postfix({'batch loss': loss, 'learning rate': lr})
        
        average_loss = average_loss / train_loader.__len__()
        train_loss.append(average_loss)
        print(f"Train Epoch {i}: average loss={average_loss}, learning rate={lr}")
        
        print(f"Evaluating with validation dataset...")
        val_loss, val_accuracy = validation(Model, val_dataloader=val_loader)
        _, train_accuracy = validation(Model, val_dataloader=train_accuracy_loader)
        # _, test_accuracy = validation(Model, val_dataloader=test_loader)
        train_accuracy_buffer.append(train_accuracy)
        validation_loss.append(val_loss)
        validation_accuracy.append(val_accuracy)
        # test_accuracy_list.append(test_accuracy)
        print(f"Evaluating done, validation loss = {val_loss}, validation accuracy = {100*val_accuracy:.2f}%")# , train accuracy = {100*train_accuracy:.2f}%, test accuracy = {100*test_accuracy:.2f}%")
        
        # save best model according to validation
        if val_accuracy > best_accuracy_val:
            Model.save("Model.pkl")
            best_accuracy_val = val_accuracy
        
        # learning rate decay
        if (i+1) % 10 == 0:
            lr_schechlar.step()
        # cos_lr_scheduler.step()
    print(f"During training process, best accuracy at validation set is {100*best_accuracy_val:.2f}%")
    # plot train loss and validation loss
    epoch = np.arange(1, len(train_loss)+1)
    plt.figure()
    plt.plot(epoch, train_loss, label='Training loss', color='green')
    plt.plot(epoch, validation_loss, label='Validation loss', color='blue')
    plt.title('Training and validation loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig("Training_validation_loss.jpg")

    plt.figure()
    validation_accuracy = [i * 100 for i in validation_accuracy]
    train_accuracy_buffer = [i * 100 for i in train_accuracy_buffer]
    # test_accuracy_list = [i * 100. for i in test_accuracy_list]
    plt.plot(epoch, validation_accuracy, label='Validation accuracy', color='blue')
    plt.plot(epoch, train_accuracy_buffer, label='Training accuracy', color='green')
    # plt.plot(epoch, test_accuracy_list, label='Test accuracy', color='yellow')
    plt.title('Train and validation accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.legend()
    plt.grid()
    plt.savefig("Training_validation_accuracy.jpg")

