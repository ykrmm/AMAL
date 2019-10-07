# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate
"""
    Auteur : Yannis Karmim Marc Treu 
"""
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from datamaestro import prepare_dataset 
from torch.utils import data

class Boston_Dataset(data.Dataset):
    def __init__(self,file_path="/Users/ykarmim/Documents/Cours/Master/M2/AMAL/TME/tme1/housing.data"):
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



class Context:
    """Very simplified context object"""
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors



class MaFonction(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx,grad_output):
        input = ctx.saved_tensors
        return grad_output.mm(input)


## Exemple d'implementation de fonction a 2 entrÃ©es
class MaFonction(Function):
    @staticmethod
    def forward(ctx,x,w):
        ## Calcul la sortie du module
        ctx.save_for_backward(x,w)
        return x.mm(w.t())

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrÃ©es
        x,w = ctx.saved_tensors
        return grad_output.mm(w), grad_output.t().mm(x)


## Pour utiliser la fonction 
#mafonction = MaFonction()
#ctx = Context()
#output = mafonction.forward(ctx,x,w)
#mafonction_grad = mafonction.backward(ctx,1)

## Pour tester le gradient 
mafonction_check = MaFonction.apply
x = torch.randn(10,5,requires_grad=True,dtype=torch.float64)
w = torch.randn(1,5,requires_grad=True,dtype=torch.float64)
print(torch.autograd.gradcheck(mafonction_check,(x,w)))

## Pour telecharger le dataset Boston
ds=prepare_dataset("edu.uci.boston")
fields, data =ds.files.data() 