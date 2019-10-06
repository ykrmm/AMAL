#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:14:23 2019

@author: 3775070
"""


from datamaestro import prepare_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class Mnist_dataset(Dataset):

    def __init__(self, X, y):
        self.data = torch.from_numpy(X/255)
        self.labels = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class Autoencodeur(torch.nn.Module):

    def __init__(self, d_in, d_h):
        super(Autoencodeur, self).__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(d_in, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, d_h)
        )
        self.decode = torch.nn.Sequential(
            torch.nn.Linear(d_h, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, d_in),
        )

        self.linear_encode = torch.nn.Linear(d_in, d_h)
        self.linear_decode = torch.nn.Linear(d_h, d_in)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.Sigmoid()

    def foward(self, x):

        y = self.linear_encode(x)
        y = self.act1(y)
        y = self.linear_encode(y)
        y = self.act2(y)
        y = self.linear_decode(y)

        return y

if __name__ == '__main__':

    ds = prepare_dataset("com.lecun.mnist")
    train_images ,  train_labels = ds.files["train/images" ].data(), ds.files["train/labels"].data()
    test_images ,  test_labels = ds.files["test/images"].data(), ds.files["test/labels"].data()
    dataset_train = Mnist_dataset(train_images, train_labels)
    dataset_test = Mnist_dataset(test_images,test_labels)
    
    
    batch_size = 10
    train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset_test,shuffle=True,batch_size=dataset_test.__len__())
    learning_rate= 10e-4
    model = Autoencodeur(28*28,20)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    epoch = 10
    for i in range(epoch):
        for (x, y) in enumerate(train_loader):
            pass
