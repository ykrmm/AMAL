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
# Loss reduction = None
# colalte_fn pour avoir des tailles fixes de batch. 
# On doit mettre une fin de séquence <eos> caractère à prendre compte dans l'apprentissage.

class Trump_dataset(Dataset):

    def __init__(self, file ="trump_full_speech.txt",dim_embeddings=300):
        """
            keep_n_columns : number of town we want to keep to train our model
        """
        f = open("trump_full_speech.txt","r").read().split() 
        dict_word = {}

        for word in f: 
            dict_word.setdefault(word,torch.zeros(dim_embeddings)) # on veut des lettres


    def normalize(s):
        return ''.join(c for c in unicodedata.normalize('NFD',s) if c in LETTRES)

    def string2code(s):
        return torch.tensor([lettre2id[c] for c in normalize(s)])
    def code2string(t):
        if type(t) != list:
            t = t.tolist()
        return ''.join(id2lettre[i] for i in t)
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
                    print('av')
                    batch_data.append(self.data_train[ind_ville][debut-self.seq_length:debut])
                    print(batch_data)
                else:
                    batch_data.append(self.data_train[ind_ville][debut:debut+self.seq_length])
                batch_label.append(self.labels[ind_ville])
            all_batch_train_data.append(batch_data)
            all_batch_train_labels.append(batch_label)
        self.all_batch_train_data = torch.Tensor(all_batch_train_data)
        self.all_batch_train_labels = torch.Tensor(all_batch_train_labels)
        
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
                batch_label.append(self.labels[ind_ville])
            all_batch_test_data.append(batch_data)
            all_batch_test_labels.append(batch_label)
        self.all_batch_test_data = torch.Tensor(all_batch_test_data)
        self.all_batch_test_labels = torch.Tensor(all_batch_test_labels)

        self.all_batch_test=(self.all_batch_test_data,self.all_batch_test_labels)

        return self.all_batch_train,self.all_batch_test
        


    


class MY_LSTM(torch.nn.Module):

    def __init__(self, dimx, dimh,batch_size):
        # dim x = embeddings shape
        super(MY_LSTM, self).__init__()
        self.Wf = torch.nn.Linear(dimx+dimh,dimh)
        self.Wi = torch.nn.Linear(dimx+dimh,dimh)
        self.Wo = torch.nn.Linear(dimx+dimh,dimh)
        self.Wc = torch.nn.Linear(dimx+dimh,dimh)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.Ct = torch.zeros(dimh)
        self.ht = torch.zeros(dimh)
        self.h = torch.zeros(dimh)

    def one_step(self,x):
        x_ht = torch.cat((self.ht,x),0)
        ft = self.sigmoid(self.Wf(x_ht))
        it = self.sigmoid(self.Wi(x_ht))
        self.Ct = ft*self.Ct + it * self.tanh(self.Wc(x_ht))
        ot = self.sigmoid(self.Wo(x_ht))
        self.ht = ot * self.tanh(self.Ct) # Est ce qu'on doit garder en memoire h ? 
        self.h = self.h.cat((self.h,self.ht))

    def forward(self,seq):
        for x in seq:
            self.one_step(x)
        
        return self.h


if __name__ == '__main__':
    


    dataset = Temperature_dataset(seq_length=30)
    batch_size = 50
    batch_train,batch_test = dataset.construct_batch(batch_size=batch_size)
    writer = SummaryWriter()

    savepath = "save_net/rnn_temperature.model"

    dim_h = 20
    dim_x = 1 
    model = RNN(dim_x,dim_h,batch_size)
     # Sinon bug... Jsp pourquoi
    learning_rate= 10e-4 
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #state = State(model,optimizer)    
    criterion = torch.nn.CrossEntropyLoss()
    epoch = 100
    print(" ------------ ENTRAINEMENT RESEAU DE NEURONES ---------------")
    for ep in range(epoch):
        print("EPOCHS : ",ep)
        for i, (x, y) in enumerate(batch_train):
            model.train()
            pred = model(x)
            #print(pred)
            loss = criterion(pred.double(), x.double())
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
