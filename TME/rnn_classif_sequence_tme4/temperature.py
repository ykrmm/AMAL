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
#############
class Ville:
    def __init__(self,path_csv_ville = "data/city_attributes.csv"):
        self.ville_df = pd.read_csv(path_csv_ville)

    def get_df(self):
        return self.ville_df



class Temperature_dataset(Dataset):

    def __init__(self, path_temp_csv ="data/tempAMAL_train.csv"):

        self.temp_data_df = pd.read_csv(path_temp_csv)
        self.list_label = self.temp_data_df.columns[1:].values
        self.temp_data_df = self.temp_data_df.fillna(self.temp_data_df.mean()) # remplace NaN
        data =[]
        labels = []
        for ville in self.list_label:
            y = list((ville==self.list_label).astype(int))
            for x in list(self.temp_data_df[ville].values):
                data.append(x)
                labels.append(y) # Encodage one-hot des classes
        self.data = torch.Tensor(data)
        #self.data = self.data.double()
        self.labels = torch.Tensor(labels)

    def __getitem__(self, index):

        return self.data[index], self.labels[index]

    def __len__(self):

        return len(self.labels)

    


class RNN(torch.nn.Module):

    def __init__(self, d_in_x, d_h,batch_size):

        super(RNN, self).__init__()
        self.W = torch.nn.Linear(d_in_x + d_h,d_h)
        self.activ = torch.nn.Tanh()
        self.batch_size = batch_size
        self.d_h = d_h

    def one_step(self,x):
        #print("x",x.size())
        #print("h",self.h[self.t].size())
        self.h.append(self.activ(self.W((torch.cat((self.h[self.t],x.view(1,self.batch_size)),0)).T)))

    def forward(self,seq):
        self.h = [torch.zeros((self.d_h,self.batch_size))]
        self.t = 0
        #print(seq)
        
        self.one_step(seq)
        self.t+=1
        print(self.h)
        print(self.h[0])
        self.h = torch.Tensor(self.h) # Problème sur une liste de tenseurs
        
        return self.h


if __name__ == '__main__':
    


    dataset = Temperature_dataset()
    batch_size = 57 
    
    """array([    1,     3,     5,     9,    13,    15,    19,    39,    45,
          57,    65,    95,   117,   171,   195,   247,   285,   585,
         741,   855,  1235,  2223,  3705, 11115]) Choisir une de ces batch_size pour éviter de prendre des séquences diff"""
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
            
        """  
        for i,(x,y) in enumerate(test_loader):
            with torch.no_grad():
                model.eval()
                
                pred = model(x)
                loss = criterion(pred.double(),x.double())
                # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
                writer.add_scalar('Loss/test', loss, ep)"""
    try:
        torch.save(model.state_dict(), savepath)
        print("model successfully saved in",savepath)
    except:
        print("something wrong with torch.save(model.state_dict(),savepath)")






# À FAIRE : SPLIT TRAIN EN DEUX FICHIER TRAIN ET TEST
