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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np
import random
#############
"""
Pour chaque cellule de notre réseau récurrent on veut pouvoir prédire l'élement suivant de notre séquence, qui est ici une
température. On veut donc pouvoir à chaque pas de temps t et sachant une  séquence de température, prédire la température suivante. 

La fonction de loss la plus adapté est donc la MSE, puisqu'on est plus dans un problème de classification, mais plus de régression.

Idée : 
Plot target et les prédictions pour les températures afin de comparer en plus de la loss MSE.
"""

class Ville:
    def __init__(self,path_csv_ville = "data/city_attributes.csv"):
        self.ville_df = pd.read_csv(path_csv_ville)

    def get_df(self):
        return self.ville_df



class Temperature_dataset:

    def __init__(self, path_temp_csv ="data/tempAMAL_train.csv",keep_n_columns=5,seq_length=30,test_size=0.2):
        """
            keep_n_columns : number of town we want to keep to train our model
        """
        # Il faut normaliser les températures.
        self.seq_length = seq_length
        self.temp_data_df = pd.read_csv(path_temp_csv)
        print(self.temp_data_df.columns)
        self.temp_data_df.drop('datetime',inplace=True,axis=1) # delete datetime
        self.temp_data_df.drop(self.temp_data_df.columns[keep_n_columns:],inplace=True,axis=1) # keep the n first columns in the dataset
        self.list_label = self.temp_data_df.columns.values # array labels 
        self.temp_data_df = self.temp_data_df.fillna(self.temp_data_df.mean()) # remplace NaN
        
        self.train_df, self.test_df = train_test_split(self.temp_data_df, test_size=test_size) # Split train_test
        self.data_train =[]
        self.labels = []
        for ville in self.list_label:
            self.labels.append(list((ville==self.list_label).astype(int)))
            self.data_train.append(list(self.train_df[ville].values))
        #self.data_train = torch.Tensor(data_train)
        
        #self.labels = torch.Tensor(labels) # meme labels train test relié aux columns des data
        
        self.data_test =[]
        for ville in self.list_label:
            self.data_test.append(list(self.test_df[ville].values))
        #self.data_test = torch.Tensor(data_train)


    def construct_batch(self, batch_size = 50):
        all_batch_train_data = []
        all_batch_train_labels = []
        for _ in range(len(self.data_train[0])//10):
            batch_data = []
            batch_label = []
            for _ in range(batch_size):
                ind_ville = np.random.randint(len(self.labels))
                debut = np.random.randint(len(self.data_train[0]))
                if  len(self.data_train[ind_ville]) - debut < self.seq_length:
                    batch_data.append(self.data_train[ind_ville][debut-self.seq_length:debut])                
                else:
                    batch_data.append(self.data_train[ind_ville][debut:debut+self.seq_length])
                #batch_label.append(self.labels[ind_ville]) encodage one_hot
                batch_label.append(ind_ville) # on donne les indices de la ville directement, requis pour Cross Entropy
            all_batch_train_data.append(batch_data)
            all_batch_train_labels.append(batch_label)
        self.all_batch_train_data = torch.Tensor(all_batch_train_data)
        self.all_batch_train_labels = torch.Tensor(all_batch_train_labels)
        #self.all_batch_train_data = self.all_batch_train_data.double()
        #self.all_batch_train_labels = self.all_batch_train_labels.double()
        
        self.all_batch_train=(self.all_batch_train_data,self.all_batch_train_labels)
        
        all_batch_test_data = []
        all_batch_test_labels = []
        for _ in range(len(self.data_test[0])//10):
            batch_data = []
            batch_label = []
            for _ in range(batch_size):
                ind_ville = np.random.randint(len(self.labels))
                debut = np.random.randint(len(self.data_test[ind_ville]))
                if  len(self.data_test[ind_ville]) - debut < self.seq_length:
                    batch_data.append(self.data_test[ind_ville][debut-self.seq_length:debut])
                else:
                    batch_data.append(self.data_test[ind_ville][debut:debut+self.seq_length])
                #batch_label.append(self.labels[ind_ville]) encodage one hot
                batch_label.append(ind_ville)
            all_batch_test_data.append(batch_data)
            all_batch_test_labels.append(batch_label)
        self.all_batch_test_data = torch.Tensor(all_batch_test_data)
        self.all_batch_test_labels = torch.Tensor(all_batch_test_labels)
        #self.all_batch_test_data = self.all_batch_test_data.double()
        #self.all_batch_test_labels = self.all_batch_train_labels.double()

        self.all_batch_test=(self.all_batch_test_data,self.all_batch_test_labels)

        return self.all_batch_train,self.all_batch_test
        


    


class RNN(torch.nn.Module):

    def __init__(self, d_in_x, d_h,d_out,batch_size):
        super(RNN, self).__init__()
        self.Wh = torch.nn.Linear(d_h,d_h) # Couche de la représentation latente h 
        self.Wx = torch.nn.Linear(d_in_x,d_h) # Couche entrante de X
        self.Wo = torch.nn.Linear(d_h,d_out) # Couche de sortie (celle qui va prédire la température)
        self.activ = torch.nn.Tanh() # Activation entre les h, on utilise pas de couche d'activation pour la sortie. 
        self.batch_size = batch_size
        self.d_h = d_h
        self.d_in_x = d_in_x
        self.d_out = d_out
        #self.ht = torch.zeros((batch_size,d_h),requires_grad=True,dtype=torch.double) # representation h courante
        #self.h = torch.zeros((batch_size,d_h),requires_grad=True,dtype=torch.double) # tout nos h gardés en mémoire
        #self.ot = torch.zeros((batch_size,d_out),requires_grad=True,dtype=torch.double) # La sortie courante notre température prédit
        #self.o = torch.zeros((batch_size,d_out),requires_grad=True,dtype=torch.double) # Toute nos sorties gardées en mémoire.
        

    def one_step(self,x):
        x = x.double()
        self.x = x
        x = x.view((self.batch_size,self.d_in_x))
        s1 = self.Wh(self.ht)
        s2 = self.Wx(x)
        self.ht = self.activ(s1 +s2) # activation entre nos cellules du réseau 
        self.h = torch.cat((self.h,self.ht),1)
        self.ot = self.Wo(s1 + s2) # pas d'activation pour la sortie (regression)
        self.o = torch.cat((self.o,self.ot),1)

    def forward(self,seq):
        
        self.ht = torch.zeros((self.batch_size,self.d_h),requires_grad=True,dtype=torch.double) # representation h courante
        self.h = torch.zeros((self.batch_size,self.d_h),requires_grad=True,dtype=torch.double) # tout nos h gardés en mémoire 
        self.ot = torch.zeros((self.batch_size,self.d_out),requires_grad=True,dtype=torch.double) # La sortie courante notre température prédit
        self.o = torch.zeros((self.batch_size,self.d_out),requires_grad=True,dtype=torch.double) # Toute nos sorties gardées en mémoire
        for x in range(len(seq[0])):
            self.one_step(seq[:,x])

        return self.o


if __name__ == '__main__':
    


    dataset = Temperature_dataset(seq_length=30)
    batch_size = 100
    batch_train,batch_test = dataset.construct_batch(batch_size=batch_size)
    writer = SummaryWriter()

    savepath = "savenet/rnn_pred_temp.model"

    dim_h = 10 # Taille des représentations internes des cellules du réseaux
    dim_o = 1 # Dimension de sortie du réseau 
    dim_x = 1 
    model = RNN(d_in_x=dim_x,d_h=dim_h,d_out=dim_o,batch_size=batch_size)
    model = model.double()
     # Sinon bug... Jsp pourquoi
    learning_rate= 10e-4 
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #state = State(model,optimizer)    
    criterion = torch.nn.MSELoss()
    epoch = 70
    print(" ------------ ENTRAINEMENT RESEAU DE NEURONES ---------------")
    for ep in range(epoch):
        print("EPOCHS : ",ep)
        data,label = batch_train
        for i, x in enumerate(data):
            model.train()
            pred = model(x)
            y = x.T[1:].T
            pred = pred.T[1:].T 
            pred = pred.T[:len(pred.T)-1].T
            loss = criterion(pred, y.double())

            a=np.random.randint(len(y)) # On prend aléatoirement une température du pred et du y pour comparer
            b = np.random.randint(len(y[0]))

            writer.add_scalar('Loss/train_forecast', loss, ep)
            writer.add_scalars(f'Loss/train_compare_temp', {'pred_temp': pred[a,b],'true_temp': y[a,b],}\
                , ep)


            if ep==0:
                loss.backward(retain_graph=True)
            else:
                try:
                    loss.backward()
                except:
                    loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            print(str(i)+'/'+str(len(data)))
            
        data,label = batch_test
        for i,x in enumerate(data):
            with torch.no_grad():
                model.eval()             
                pred = model(x)
                y = x.T[1:].T
                pred = pred.T[1:].T 
                pred = pred.T[:len(pred.T)-1].T
                loss = criterion(pred,y.double())

                a=np.random.randint(len(y)) # On prend aléatoirement une température du pred et du y pour comparer
                b = np.random.randint(len(y[0]))

                writer.add_scalar('Loss/test_forecast', loss, ep)
                writer.add_scalars(f'Loss/test_compare_temp', {'pred_temp': pred[a,b],'true_temp': y[a,b],}\
                , ep)

    try:
        torch.save(model.state_dict(), savepath)
        print("model successfully saved in",savepath)
    except:
        print("something wrong with torch.save(model.state_dict(),savepath)")


