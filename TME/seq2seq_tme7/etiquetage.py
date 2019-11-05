#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:14:23 2019

auteurs : Yannis Karmim & Marc Treu
"""

import itertools
import logging
from tqdm import tqdm
import unicodedata
import string

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pandas as pd
from datamaestro import prepare_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from torch.autograd import gradcheck
import torch.nn.utils.rnn as rnn
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


### Partie Tagging

logging.basicConfig(level=logging.INFO)

from datamaestro import prepare_dataset
ds = prepare_dataset('org.universaldependencies.french.gsd')

BATCH_SIZE=100

# Format de sortie
# https://pypi.org/project/conllu/

class VocabularyTagging:
    OOVID = 0

    def __init__(self, oov: bool):
        self.oov =  oov
        if oov:
            self.word2id = { "__OOV__": VocabularyTagging.OOVID }
            self.id2word = [ "__OOV__" ]
        else:
            self.word2id = {}
            self.id2word = []
    
    def __getitem__(self, i):
        return self.id2word[i]


    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return VocabularyTagging.OOVID
            raise


    def __len__(self):
        return len(self.id2word)


class TaggingDataset(Dataset):
    def __init__(self, data, words: VocabularyTagging, tags: VocabularyTagging, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, ix):
        return self.sentences[ix]

    @staticmethod
    def collate_fn(batch):
        # Renvoyer séquence paddé + longueur des séquences
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return (rnn.pad_sequence(data, batch_first=True), rnn.pad_sequence(labels, batch_first=True),[len(i) for i in data])




class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence[0])
        pack_sequence = rnn.pack_padded_sequence(embeds,sentence[2],batch_first=True,enforce_sorted=False)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores



logging.info("Loading datasets...")
words = VocabularyTagging(True)
tags = VocabularyTagging(False)
train_data = TaggingDataset(ds.files["train"], words, tags, True)
dev_data = TaggingDataset(ds.files["dev"], words, tags, True)
test_data = TaggingDataset(ds.files["test"], words, tags, False)


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,collate_fn=TaggingDataset.collate_fn)
dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE,collate_fn=TaggingDataset.collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,collate_fn=TaggingDataset.collate_fn)
logging.info("Vocabulary size: %d", len(words))


# Loss sur packpadded.data des data et des labels

model = LSTMTagger()
learning_rate=10e-3
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

writer = SummaryWriter()

n_epoch = 50

print(" ------------ ENTRAINEMENT RESEAU DE NEURONES ---------------")
for ep in range(n_epoch):
    print("EPOCHS : ",ep)
    for i, (x, y) in enumerate(train_loader):
        model.train()
        y = y#.float()
        x = x#.double()
        
        pred = model(x)
        print(pred)
        loss = criterion(pred, y)
        # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
        writer.add_scalar('Loss/train', loss, ep)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    for i,(x,y) in enumerate(test_loader):
        with torch.no_grad():
            model.eval()
            pred = model(x)
            loss = criterion(pred,y)
            # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
            writer.add_scalar('Loss/validation', loss, ep)


"""

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






# À FAIRE : SPLIT TRAIN EN DEUX FICHIER TRAIN ET TEST

"""