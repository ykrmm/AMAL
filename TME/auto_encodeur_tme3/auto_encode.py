#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:14:23 2019

auteurs : Yannis Karmim & Marc Treu
"""


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

class Mnist_dataset(Dataset):

    def __init__(self, X, y):

        liste_x= []
        self.labels = torch.from_numpy(y)#.double()
        data = X/255#.double()
        for d in data:
            liste_x.append(d.reshape((784,))) # On met sous forme de vecteur sinon la shape 28x28 ne passe pas
        
        self.data = torch.Tensor(liste_x)
        self.data = self.data.double()

    def __getitem__(self, index):

        return self.data[index], self.labels[index]

    def __len__(self):

        return len(self.labels)

    def re_normalize(self):

        self.data*=255
    
    def compare_images(self,index,prediction,save=False,fname=None):

        """
            Compare our predictions with matplotlib and save it. 
            prediction -> Tensor 
            index -> int to use the __getitem__ function.
        """
        xtrue = self.__getitem__(index)[0]
        xtrue = xtrue * 255 # On remet des valeurs entre 0 et 255 pour l'affichage
        xtrue = xtrue.view((28,28)) # On remet au format 28x28 pixels
        prediction = prediction*255
        prediction = prediction.view((28,28))
        _, axarr = plt.subplots(2,sharex=True,sharey=True)
        axarr[0].set_title('original image')
        axarr[0].imshow(xtrue)
        axarr[1].set_title('output autoencoder')
        axarr[1].imshow(prediction)
        if fname is not None :
            path = os.path.join('figures',fname)
            plt.savefig(path)


class Autoencodeur(torch.nn.Module):

    def __init__(self, d_in, d_h):

        super(Autoencodeur, self).__init__()
        self.linear_encode = torch.nn.Linear(d_in, d_h)
        self.linear_decode = torch.nn.Linear(d_h, d_in)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.Sigmoid()


    def forward(self, x):

        y = self.linear_encode(x)
        y = self.act1(y)
        w = self.linear_encode.weight
        #b = self.linear_encode.bias
        w = torch.nn.Parameter(w.t())
        self.linear_decode.weight = w
        #self.linear_decode.bias = b
        y = self.linear_decode(y) # il faut lui forcer à utiliser les mêmes weight et biais que l'encodeur 
        y = self.act2(y)

        return y
"""
class State: 
    def __init__(self,model,optim):
        self.model = model
        self.optim = optim
        self.epoch , self.iteration = 0,0
"""


if __name__ == '__main__':
    

    ds = prepare_dataset("com.lecun.mnist")
    train_images ,  train_labels = ds.files["train/images" ].data(), ds.files["train/labels"].data()
    test_images ,  test_labels = ds.files["test/images"].data(), ds.files["test/labels"].data()

    dataset_train = Mnist_dataset(train_images, train_labels)
    dataset_test = Mnist_dataset(test_images,test_labels)
    batch_size = 65
    train_loader = DataLoader(dataset_train,shuffle=True,batch_size=batch_size)
    test_loader = DataLoader(dataset_test,shuffle=True,batch_size=batch_size)

    writer = SummaryWriter()

    savepath = "save_net/auto_encoder.model"

    
    model = Autoencodeur(784,120)
    model = model.double() # Sinon bug... Jsp pourquoi
    learning_rate= 10e-4 
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #state = State(model,optimizer)    
    criterion = torch.nn.BCELoss()
    epoch = 30
    print(" ------------ ENTRAINEMENT RESEAU DE NEURONES ---------------")
    for ep in range(epoch):
        print("EPOCHS : ",ep)
        for i, (x, y) in enumerate(train_loader):
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


    # Affichage image
    index = random.randint(0,len(test_images))
    x_to_pred = dataset_test.data[index]
    with torch.no_grad():
        model.eval()
        pred = model(x_to_pred)
    dataset_test.compare_images(index,pred,save=True,fname='test5.png')