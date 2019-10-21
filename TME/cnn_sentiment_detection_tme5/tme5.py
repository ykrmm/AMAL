#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:14:23 2019

auteurs : Yannis Karmim & Marc Treu
"""

import pandas as pd
from datamaestro import prepare_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from torch.autograd import gradcheck
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import tp5_preprocess as tp5
import gzip

class CNN(torch.nn.Module):

    def __init__(self, d_in_x, d_h,batch_size):
        super(CNN, self).__init__()

    def forward(self,seq):
        pass

class TextDataset(torch.utils.data.Dataset):
    
    def __init__(self, text: torch.LongTensor, sizes: torch.LongTensor, labels: torch.LongTensor):
        self.text = text
        self.sizes = sizes
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.text[self.sizes[index]:self.sizes[index+1]], self.labels[index].item()

    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return Batch(torch.nn.utils.rnn.pad_sequence(data, batch_first=True), torch.LongTensor(labels))


if __name__ == '__main__':
    

    with gzip.open('train-1000.pth') as fp:
        train_ds = torch.load(fp)
    print(train_ds)


    data_loader = DataLoader(train_ds,shuffle=False,batch_size=10)

    for i, x in enumerate(data_loader):
        print(x)
        break






    """
    dataset = Temperature_dataset()
    batch_size = 57 
    
    data_loader = DataLoader(dataset,shuffle=False,batch_size=batch_size)
    writer = SummaryWriter()

    savepath = "save_net/rnn_temperature.model"

    dimx =  1
    dimh = 30
    model = RNN(dimx,dimh,batch_size)
     # Sinon bug... Jsp pourquoi
    learning_rate= 10e-4 
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #state = State(model,optimizer)    
    criterion = torch.nn.CrossEntropyLoss()
    epoch = 100
    print(" ------------ ENTRAINEMENT RESEAU DE NEURONES ---------------")
    for ep in range(epoch):
        print("EPOCHS : ",ep)
        for i, (x, y) in enumerate(data_loader):
            model.train()
            pred = model(x).double()
            #print(pred)
            loss = criterion(pred.double(), x.double())
            # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
            writer.add_scalar('Loss/train', loss, ep)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    
        for i,(x,y) in enumerate(test_loader):
            with torch.no_grad():
                model.eval()
                
                pred = model(x)
                loss = criterion(pred.double(),x.double())
                # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
                writer.add_scalar('Loss/test', loss, ep)
    try:
        torch.save(model.state_dict(), savepath)
        print("model successfully saved in",savepath)
    except:
        print("something wrong with torch.save(model.state_dict(),savepath)")
    """





    # À FAIRE : SPLIT TRAIN EN DEUX FICHIER TRAIN ET TEST
