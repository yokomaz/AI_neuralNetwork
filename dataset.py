import pickle
import numpy as np
import random

class Dataset:
    def __init__(self, train, test): # ,batch_size
        # self.batch_size = batch_size
        self.train = train
        self.test = test

    @classmethod
    def load(cls, dir_dataset):
        train = {}
        test = {}

        # load train data
        train_label = []
        train_data = []
        for i in range(1, 6):
            with open(f"{dir_dataset}/data_batch_{i}", "rb") as file:
                dict = pickle.load(file, encoding='bytes')
            train_label += dict[b'labels']
            train_data.append(dict[b'data'])
        train_data = np.vstack(train_data)
        train_label = np.asarray(train_label).reshape(-1, 1)
        # standerize data
        # train_data_mean = np.mean(train_data, axis=1, keepdims=True)
        # train_data_std = np.std(train_data, axis=1, keepdims=True)
        # train_data = (train_data - train_data_mean) / train_data_std
        train_data = train_data / 255
        train['data'] = train_data
        train['label'] = train_label

        # load test data
        # test_label = []
        # test_data = []
        with open(f"{dir_dataset}/test_batch", "rb") as file:
            dict = pickle.load(file, encoding='bytes')
        test_label = np.asarray(dict[b'labels']).reshape(-1, 1)
        test_data = np.asarray(dict[b'data'])
        # standerize data
        # test_data_mean = np.mean(test_data, axis=1, keepdims=True)
        # test_data_std = np.std(test_data, axis=1, keepdims=True)
        # test_data = (test_data - test_data_mean) / test_data_std # normalization
        test_data = test_data / 255
        test['data'] = test_data
        test['label'] = test_label

        return cls(train, test)
    
    def __getitem__(self, idx):
        return self.train['data'][idx], self.train['label'][idx]

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        self.indices = np.arange(self.num_samples)
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            batch_idx = self.indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]
            
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size