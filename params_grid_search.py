import numpy as np
from model import neuralNetwork#, neuralNetwork_v2
from dataset import Dataset, DataLoader
from utils import one_hot_encode, lr_scheduler
from sklearn.model_selection import train_test_split
from loss_function import cross_entropy, validation
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr_scheduler = lr

    def step(self):
        # for param in self.params:
        lr = self.lr_scheduler.get_lr()
        self.params['input_layer_weights'] -= lr * self.params['input_layer_weights_grad']
        self.params['input_layer_bias'] -= lr * self.params['input_layer_bias_grad']
        self.params['hidden_layer_weights'] -= lr * self.params['hidden_layer_weights_grad']
        self.params['hidden_layer_bias'] -= lr * self.params['hidden_layer_bias_grad']
        self.params['output_layer_weights'] -= lr * self.params['output_layer_weights_grad']
        self.params['output_layer_bias'] -= lr * self.params['output_layer_bias_grad']
    
    def zero_grad(self):
        # for param in self.params:
        self.params['input_layer_weights_grad'] = 0
        self.params['input_layer_bias_grad'] = 0
        self.params['hidden_layer_weights_grad'] = 0
        self.params['hidden_layer_bias_grad'] = 0
        self.params['output_layer_weights_grad'] = 0
        self.params['output_layer_bias_grad'] = 0

if __name__=="__main__":
    # hyperparameters
    nb_epoch = 200
    learning_rate = [1e-2, 1e-3, 1e-4]
    activation = ['relu'] #['relu', 'sigmoid'] # 'sigmoid'
    l2_lambda = [1e-3] # 1e-4
    hidden_dim = [256] # 1536
    batch_size = [200] # 100

    params_grid = list(itertools.product(learning_rate, activation, l2_lambda, hidden_dim, batch_size))
    print(f"Length of params grids is {len(params_grid)}")

    # prepare dataset and do some preprocessing
    print(f"Preparing dataset")
    dataset = Dataset.load("dataset_cifar10")
    train_set = dataset.train
    # test_set = dataset.test
    
    x_train, x_val, y_train, y_val = train_test_split(train_set['data'], train_set['label'], random_state=42, test_size=0.2, shuffle=True, stratify=train_set['label'])
    print(f"Done")
    
    print(f"Convering labels to one-hot-encode")
    y_train = one_hot_encode(y_train)
    y_val = one_hot_encode(y_val)
    # test_set['label'] = one_hot_encode(test_set['label'])
    # x_test = test_set['data']
    # y_test = test_set['label']

    # start training
    best_accuracy = 0
    plt.figure()
    print(params_grid)
    best_grid = None

    for (learn_rate, activate, l2_lamb, hd_dim, bs) in params_grid:
        train_loader = DataLoader(x_train, y_train, batch_size=bs)
        val_loader = DataLoader(x_val, y_val, batch_size=len(y_val))
        # test_loader = DataLoader(x_test, y_test, batch_size=len(y_test))

        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]
        # params_grid.append([learn_rate, activate, l2_lamb, hd_dim, bs])
        print(f"Learning rate {learn_rate}, activation {activate}, l2_lambda {l2_lamb}, hidden dim {hd_dim}, batch size = {bs}")
        lr_schech = lr_scheduler(learn_rate, gamma=0.98)
        Model = neuralNetwork(input_dim=input_dim, hidden_dim=hd_dim, output_dim=output_dim, activate=activate)
        optim = SGD(Model.parameters, lr=lr_schech)
        
        # 对于每一个grid参数，都要有记录其train loss，validation loss，validation accuracy的数组\
        # 方便之后进行绘图
        train_loss = []
        validation_loss = []
        validation_accuracy = []
        # test_accuracy_list = []
        accuracy_loss_best = 0

        for i in range(nb_epoch):
            lr = lr_schech.get_lr()
            average_loss = 0
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, colour='green')
            for batch_idx, (batch_x, batch_y) in loop:
                y_pred = Model.forward(batch_x)                
                loss = cross_entropy(y_pred, batch_y, Model, l2_lambda=l2_lamb)
                average_loss += loss
                Model.backward(batch_y, l2_lamb, learning_rate=lr)
                optim.step()
                optim.zero_grad()

                loop.set_description(f'Train Epoch: [{i+1}/{nb_epoch}]')
                loop.set_postfix({'batch loss': loss, 'learning rate': lr})
            
            average_loss = average_loss / train_loader.__len__()
            train_loss.append(average_loss)
            print(f"Train Epoch {i}: average loss={average_loss}, learning rate={lr}")
            
            
            print(f"Evaluating with validation dataset...")
            val_loss, val_accuracy = validation(Model, val_dataloader=val_loader)
            # _, test_accuracy = validation(Model, val_dataloader=test_loader)
            validation_loss.append(val_loss)
            validation_accuracy.append(val_accuracy)
            # test_accuracy_list.append(test_accuracy)
            print(f"Evaluating done, validation loss = {val_loss}, validation accuracy = {100*val_accuracy:.2f}%")

            if val_accuracy > accuracy_loss_best:
                accuracy_loss_best = val_accuracy
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_grid = [learn_rate, activate, l2_lamb, hd_dim, bs]
            if (i+1) %10 == 0:
                lr_schech.step()
        validation_accuracy = [i * 100 for i in validation_accuracy]
        # plot train loss and validation loss
        epoch = np.arange(1, len(train_loss)+1)
        # plt.plot(epoch, train_loss, label='Training loss', color='green')
        plt.plot(epoch, validation_accuracy, label=f"lr={learn_rate}, act_func={activate}, l2_lamb={l2_lamb}, hid_dim={hd_dim}, {bs}")
        print(f"Finished processing grid: Learning rate {learn_rate}, activation {activate}, l2_lambda {l2_lamb}, hidden dim {hd_dim}, batch size = {bs}")
    
    
    print(f"Best validation accuracy is {100 * best_accuracy:.2f}, params grid = {best_grid}")
    plt.title('Validation accuracy over epochs for different params grids')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy over validation set(%)')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("Params_grid_search.jpg")