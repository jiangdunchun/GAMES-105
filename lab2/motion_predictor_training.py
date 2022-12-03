import torch
from torch import nn
import numpy as np
import pandas as pd
import random

from IPython import display
import matplotlib
import matplotlib.pyplot as plt

import time

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def split_train_and_valid(prop, features, labels): 
    split = int(features.shape[0] * prop)

    X_valid = features[:split]
    y_valid = labels[:split]

    X_train = features[split:]
    y_train = labels[split:]

    return X_train, y_train, X_valid, y_valid

def get_accuracy(net, device, features, labels):
    features, labels = features.to(device), labels.to(device)
    y_hat = net(features)
    _, predicted = torch.max(y_hat.data, 1)
    correct_sum = (predicted == labels).sum()
    return float(correct_sum) / features.shape[0]

def load_motion_data(features_file, labels_file):
    features_data = pd.read_csv(features_file)
    labels_data = pd.read_csv(labels_file)

    train_features = torch.tensor(features_data.values, dtype=torch.float32)
    train_labels = torch.tensor(labels_data.values, dtype=torch.float32)

    return train_features, train_labels

train_features, train_labels = load_motion_data('motion_material/_features.csv', 'motion_material/_labels.csv')

def get_net_and_optimizer(device, lr):
    net = nn.Sequential(
        nn.Linear(103, 256), nn.Sigmoid(),
        nn.Linear(256, 256), nn.Sigmoid(),
        nn.Linear(256, 97))
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device=device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    return net, optimizer

loss = nn.MSELoss()


batch_size = 256
lr = 0.5
num_epochs = 20
device = try_gpu()

net, optimizer = get_net_and_optimizer(device, lr)
X_train, y_train, X_valid, y_valid = split_train_and_valid(0.05, train_features, train_labels)
for epoch in range(num_epochs):
    net.train()
    for X, y in data_iter(batch_size, X_train, y_train):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        print("loss:", l)
        l.backward()
        optimizer.step()

torch.save(net, 'motion_material/_net.pth')
#n_net = torch.load('motion_material/_net.pth')