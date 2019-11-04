#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:14:23 2019

auteurs : Yannis Karmim & Marc Treu
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from torch.autograd import gradcheck
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np
import string
import unicodedata
import warnings
warnings.filterwarnings("ignore")
#############
"""
Génération de séquence sur des données de discours de Trump.
"""

class Trump_dataset:
    def __init__(self,file_trump="data/trump_full_speech.txt"):

        with open(file_trump) as f:
            self.speech = f.read()
        self.LETTRES = string.ascii_letters+string.punctuation+string.digits+' '
        self.id2lettre = dict(zip(range(1,len(self.LETTRES)+1),self.LETTRES))
        self.id2lettre[0] = '' ##NULL CHARACTER
        self.lettre2id = dict(zip(self.id2lettre.values(),self.id2lettre.keys()))
        self.speech = self.normalize(self.speech)
        self.code_speech = self.string2code(self.speech)
        self.one_hot = self.one_hot_encoding(self.code_speech) # First we encode our char as one hot encoder type.

    def normalize(self,s):
        return ''.join(c for c in unicodedata.normalize('NFD',s)if c in self.LETTRES)

    def string2code(self,s):
        return torch.Tensor([self.lettre2id[c] for c in self.normalize(s)])

    def code2string(self,t):
        if type(t) != list:
            t = t.tolist()
        return ''.join(self.id2lettre[i]for i in t)
    def one_hot_encoding(self,code):
        one_hot = torch.zeros((len(code),len(self.id2lettre)),requires_grad=True,dtype=torch.double)
        for i in range(len(code)):
            one_hot[i][int(code[i])]=1

        return one_hot    



class Trump_Encodeur(torch.nn.Module):

    def __init__(self,vocab_size,emb_size):
        super(Trump_Encodeur,self).__init__()
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
        self.ot = self.Wo(s1 + s2) # pas d'activation pour la sortie 
        self.o = torch.cat((self.o,self.ot),1)

    def forward(self,seq):
        
        self.ht = torch.zeros((self.batch_size,self.d_h),requires_grad=True,dtype=torch.double) # representation h courante
        self.h = torch.zeros((self.batch_size,self.d_h),requires_grad=True,dtype=torch.double) # tout nos h gardés en mémoire 
        self.ot = torch.zeros((self.batch_size,self.d_out),requires_grad=True,dtype=torch.double) # La sortie courante notre température prédit
        self.o = torch.zeros((self.batch_size,self.d_out),requires_grad=True,dtype=torch.double) # Toute nos sorties gardées en mémoire
        for x in range(len(seq[0])):
            self.one_step(seq[:,x])
        return self.o




class Trump_Decodeur(torch.nn.Module):

    def __init__(self,emb_size):
        super(Trump_Decodeur,self).__init__()




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
        self.ot = self.Wo(s1 + s2) # pas d'activation pour la sortie 
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
    


    dataset = Trump_dataset()
    """
    batch_size = 100
    batch_train,batch_test = dataset.construct_batch(batch_size=batch_size)
    writer = SummaryWriter()

    savepath = "savenet/rnn_gen_trump.model"

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
    """





# À FAIRE : SPLIT TRAIN EN DEUX FICHIER TRAIN ET TEST
