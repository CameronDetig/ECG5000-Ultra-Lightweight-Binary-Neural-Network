import torch
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from prettytable import PrettyTable
import pickle

'''For loading the ECG_5000 dataset.'''

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_data(device):
    labels = []
    X = list()
    y = list()

    labels = ['0.', '1.', '2.', '3.', '4.']
    
    # Load data
    ECGdataset_train = pickle.load(open('ECG_5000_Dataset/ECG5000_train.pickle', "rb"), encoding='latin1')
    ECGdataset_val = pickle.load(open('ECG_5000_Dataset/ECG5000_validation.pickle', "rb"), encoding='latin1')

    print(f"Dataset is of type: {type(ECGdataset_train)}")
    print(f"Training Shape: {ECGdataset_train.shape}")
    print(f"Validation Shape: {ECGdataset_val.shape}")

    X_train_np = ECGdataset_train[:, 1:141] # Index 1 - 140 is features
    y_train_np = ECGdataset_train[:, 0:1] # index 0 is label
    print(f"Training Feature shape: {X_train_np.shape}, Label shape: {y_train_np.shape}")

    X_test_np = ECGdataset_val[:, 1:141] # Index 1 - 140 is features
    y_test_np = ECGdataset_val[:, 0:1] # index 0 is label
    print(f"Testing Feature shape: {X_test_np.shape}, Label shape: {y_test_np.shape}")

    # Computing mean and standard deviation of training set
    train_mean = X_train_np.mean()
    train_std = X_train_np.std()

    # Apply same transformation to both sets
    X_train_standardized = (X_train_np - train_mean) / train_std
    X_test_standardized = (X_test_np - train_mean) / train_std

    X_train_ten = torch.tensor(X_train_standardized, dtype=torch.float32)
    X_test_ten = torch.tensor(X_test_standardized, dtype=torch.float32)
    y_train_ten = torch.tensor(y_train_np, dtype=torch.long)
    y_test_ten = torch.tensor(y_test_np, dtype=torch.long)

    # Debugging: Check the shapes
    print("\nBefore Reshape:")
    print("X train Shape:", X_train_ten.shape)
    print("X test Shape:", X_test_ten.shape)
    print("y train Shape:", y_train_ten.shape)
    print("y test Shape:", y_test_ten.shape)

    # Reshape and load to device
    X_train_ten = X_train_ten.reshape((500, 1, 140)).to(device)
    X_test_ten = X_test_ten.reshape((1500, 1, 140)).to(device)
    y_train_ten = y_train_ten.reshape((500)).to(device)
    y_test_ten = y_test_ten.reshape((1500)).to(device)

    # Debugging: Check the shapes
    print("\nAfter Reshape:")
    print("X train Shape:", X_train_ten.shape)
    print("X test Shape:", X_test_ten.shape)
    print("y train Shape:", y_train_ten.shape)
    print("y test Shape:", y_test_ten.shape)

    print(X_train_ten)


    return labels, X_train_ten, X_test_ten, y_train_ten, y_test_ten


class TrainDatasets(Dataset):
    def __init__(self, x_train, y_train):
        self.len = x_train.size(0)
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.len


class TestDatasets(Dataset):
    def __init__(self, x_test, y_test):
        self.len = x_test.size(0)
        self.x_test = x_test
        self.y_test = y_test

    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index]

    def __len__(self):
        return self.len


class Loader:
    def __init__(self, batch_size, device):
        self.labels, self.x_train, self.x_test, self.y_train, self.y_test = get_data(device)
        self.batch_size = batch_size
        self.train_dataset = TrainDatasets(self.x_train, self.y_train)
        self.test_dataset = TestDatasets(self.x_test, self.y_test)

    def loader(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        return self.labels, train_loader, test_loader

    def plot_train_test_splits(self):
        table = PrettyTable()
        table.field_names = ["", "ALL", "TRAIN", "TEST", "TEST RATIO"]
        ALL_SUM, TRAIN_SUM, TEST_SUM, TEST_RATIO = 0, 0, 0, 0
        for i in range(len(self.labels)):
            TRAIN = self.y_train.tolist().count(i)
            TEST = self.y_test.tolist().count(i)
            ALL = TRAIN + TEST
            TEST_RATIO = round(TEST / ALL, 3)
            table.add_row([self.labels[i], ALL, TRAIN, TEST, TEST_RATIO])

            ALL_SUM += ALL
            TRAIN_SUM += TRAIN
            TEST_SUM += TEST
        TEST_RATIO_SUM = round(TEST_SUM / ALL_SUM, 3)
        table.add_row(['Total', ALL_SUM, TRAIN_SUM, TEST_SUM, TEST_RATIO_SUM])
        return print(table)
