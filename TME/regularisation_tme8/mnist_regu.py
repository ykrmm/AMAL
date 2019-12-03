import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

def get_dataset(batch_size, path='/tmp/datasets/mnist'):
    """
    Cette fonction charge le dataset et effectue des transformations sur chaqu
    image (listées dans `transform=...`).
    """
    train_dataset = datasets.MNIST(path, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), 
           
        ]))
    val_dataset = datasets.MNIST(path, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),       
        ]))
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader



class MNIST_nn(nn.Module):
    def __init__(self,d_in,d_h,d_out):
        super(MNIST_nn,self).__init__()
        
        self.lin1=nn.Linear(d_in,d_h)
        self.lin2=nn.Linear(d_h,d_h)
        self.lin3=nn.Linear(d_h,d_out)

        self.relu=nn.ReLU()

    def forward(self,x):
        x = x.reshape((x.shape[0],x.shape[2]*x.shape[2]))
        y = self.lin1(x)
        y = self.relu(y)

        y = self.lin2(y)
        y = self.relu(y)

        y = self.lin2(y)
        y = self.relu(y)

        y = self.lin2(y)
        y = self.relu(y)


        y = self.lin3(y)

        return y


if __name__ == "__main__":

    n_epoch = 1000
    learning_rate = 10e-3
    batch_size = 300
    train,val = get_dataset(batch_size)


    d_in,d_h,d_out = 28*28,100,10 

    model = MNIST_nn(d_in,d_h,d_out)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    writer = SummaryWriter()


    print(" ------------ ENTRAINEMENT RESEAU DE NEURONES ---------------")
    for ep in range(n_epoch):
        print("EPOCHS : ",ep)
        for i, (x, y) in enumerate(train):
            model.train()            
            pred = model(x)
            loss = criterion(pred, y)
            writer.add_scalar('Loss/train', loss, ep)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        for i,(x,y) in enumerate(val):
            with torch.no_grad():
                model.eval()
                pred = model(x)
                loss = criterion(pred,y)
                writer.add_scalar('Loss/validation', loss, ep)