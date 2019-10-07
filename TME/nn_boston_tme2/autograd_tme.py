"""
    Auteur : Yannis Karmim Marc Treu 
"""

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler # utile pour split train test 
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# y = x.w.T
# z = (y-y_pred)**2
# z.backward()
# x.grad
# w.grad 
# y.grad -> On obtient rien car variable intermediaire dans le graphe de calcul 
# Tensor(    require_grad=True) -> à partir de cette variable tensor on a besoin de calculer le gradient 
# with torch.no_grad(): (utile pour le test car on a pas  besoin du gradient)
#       Spécifie qu'on utilise plus de gradient a partir de là 

device = torch.device('cpu')
dtype = torch.float
"""
a = torch.rand((1,10),requires_grad=True)
b = torch.rand((1,10),requires_grad=True)
c = a.mm(b.t())
d = 2 * c
c.retain_grad() # on veut conserver le gradient par rapport à c
d.backward()  ## calcul du gradient et retropropagation jusqu’aux feuilles du graphe de calcul
print(d.grad) # Rien : le gradient par rapport à d n’est pas conservé
print(c.grad) # Celui-ci est conservé
print(a.grad) ## gradient de c par rapport à a qui est une feuille
print(b.grad) ## gradient de c par rapport à b qui est une feuille
"""



# #with torch.no_grad():
# #   c = a.mm(b.t())
# #    c.backward() ## Erreur


# # Examples  Descente de gradient simple: 

# x = torch.randn((1,10),requires_grad=True,dtype=dtype,device=device)
# y = torch.randint(0,2,size=(1,),dtype=dtype,device=device)

# w = torch.randn((1,10),requires_grad=True,dtype=dtype,device=device)   


# learning_rate = 10e-3

# for i in range(200):

#     y_pred = x.mm(w.T)
#     mse = (y_pred-y).pow(2)
#     mse.backward()
     
#     print("y predit :",y_pred," y_true :",y, "Loss : ",mse)
#     with torch.no_grad():
#         w -=  learning_rate*w.grad
#         # On remet le gradient à 0 pour le recalcul 
#         w.grad.zero_()



# # Même tâche mais avec le module


# x = torch.randn((1,10),requires_grad=True,dtype=dtype,device=device)
# y = torch.randint(0,2,size=(1,),dtype=dtype,device=device)


# class Fianso(torch.nn.Module): # Méthode de régression linéaire simple 
#     def __init__(self,D_in,D_out):
#         super(Fianso,self).__init__()
#         self.linear = torch.nn.Linear(D_in,D_out)
#         #self.activ = 
#     def forward(self,x):

#         y = self.linear(x)

#         return y 




# # Utilisation d'un optimiseur

# criterion = torch.nn.MSELoss()
# learning_rate = 10e-3
# model = Fianso(x.size()[1],x.size()[0])
# optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# for i in range(1000 ):

#     model.train()
#     pred = model(x)
#     loss = criterion(pred,y)

#     print('loss :',loss)
#     print("y predit :",pred," y_true :",y)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()



# Test d'un réseau à trois couches (Perceval) sur les données boston housing 


class Boston_Dataset(data.Dataset):
    def __init__(self,file_path="/Users/ykarmim/Documents/Cours/Master/M2/AMAL/TME/backward_tme1/housing.data"):
        data_ = []                                                                                                                                                                           
        labels_ = []
        lineList = [line.rstrip('\n') for line in open(file_path)]                                                                                                                               

        for i in range(len(lineList)): 
            lineList[i] = lineList[i].strip() 
            lineList[i] = ' '.join(lineList[i].split()) 
            data_.append(lineList[i].split(' '))
        for i,d in enumerate(data_) :
            labels_.append(float(d[len(d)-1]))
            data_[i] = [float(s) for s in d[:-1]] 
        self.data = torch.Tensor(data_)
        self.labels = torch.Tensor(labels_)
        
    def __getitem__(self,index):
        return self.data[index],self.labels[index]

    def __len__(self):
        return len(self.labels)
    



class Perceval(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(Perceval,self).__init__()
        self.linear1 = torch.nn.Linear(D_in,H)
        self.linear2 = torch.nn.Linear(H,H)
        self.linear3 = torch.nn.Linear(H,D_out)
        self.activ1 = torch.nn.Tanh()
        

    def forward(self,x):
        y = self.linear1(x)
        y = self.activ1(y)
        y = self.linear2(y)#.squeeze()
        y = self.activ1(y)
        y = self.linear3(y)
        return y 


batch_size = 20
shuffle_dataset = True
validation_split = .2
random_seed = 42

n_epoch = 500
learning_rate = 10e-3
dataset = Boston_Dataset()
dataloader = data.DataLoader(dataset,shuffle=shuffle_dataset,batch_size=batch_size) 


# Split train/test 

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=len(val_indices),
                                                sampler=valid_sampler)

# Script entrainement model perceval 

model = Perceval(D_in=13,H=5,D_out=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

writer = SummaryWriter()


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
        
    for i,(x,y) in enumerate(validation_loader):
        with torch.no_grad():
            model.eval()
            pred = model(x)
            loss = criterion(pred,y)
            # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
            writer.add_scalar('Loss/validation', loss, ep)




# à faire : standardisation des données : -mean/std sur le train et sur les ypred du train
#                                           en test utilisé les moyennes et ecart types calculés dans le train
# dropout
# Split train/test
print("-------------------- TEST DU RESEAU DE NEURONES --------------")

