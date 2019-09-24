import torch 
#import torch.nn.Function


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




#with torch.no_grad():
#   c = a.mm(b.t())
#    c.backward() ## Erreur


# Examples  Descente de gradient simple: 

x = torch.randn((1,10),requires_grad=True,dtype=dtype,device=device)
y = torch.randint(0,2,size=(1,),dtype=dtype,device=device)

w = torch.randn((1,10),requires_grad=True,dtype=dtype,device=device)   


learning_rate = 10e-3

for i in range(200):

    y_pred = x.mm(w.T)
    mse = (y_pred-y).pow(2)
    mse.backward()
     
    print("y predit :",y_pred," y_true :",y, "Loss : ",mse)
    with torch.no_grad():
        w -=  learning_rate*w.grad

        # On remet le gradient à 0 pour le recalcul 

        w.grad.zero_()





# Même tâche mais avec le module


x = torch.randn((1,10),requires_grad=True,dtype=dtype,device=device)
y = torch.randint(0,2,size=(1,),dtype=dtype,device=device)


class Fianso(torch.nn.Module): # Méthode de régression linéaire simple 
    def __init__(self,D_in,D_out):
        super(Fianso,self).__init__()
        self.linear = torch.nn.Linear(D_in,D_out)
        #self.activ = 
    def forward(self,x):

        y = self.linear(x)

        return y 






criterion = torch.nn.MSELoss()
learning_rate = 10e-3
model = Fianso(x.size()[1],x.size()[0])
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for i in range(1000):

    model.train()
    pred = model(x)
    loss = criterion(pred,y)

    print('loss :',loss)
    print("y predit :",pred," y_true :",y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

