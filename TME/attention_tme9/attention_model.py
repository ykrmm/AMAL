import re
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from datamaestro import prepare_dataset
import torch 
import torchvision
from torch.utils.tensorboard import SummaryWriter

EMBEDDING_SIZE = 50

ds = prepare_dataset("edu.standford.aclimdb")
word2id, embeddings = prepare_dataset('edu.standford.glove.6b.%d' % EMBEDDING_SIZE).load()

class FolderText(Dataset):
    def __init__(self, classes, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = list(classes.keys())
        for label, folder in classes.items():
            for file in folder.glob("*.txt"):
                self.files.append(file)
                if 'pos' in label:

                    self.filelabels.append(0)
                else:
                    self.filelabels.append(1)
        self.filelabels = torch.Tensor(self.filelabels)
    def __len__(self):
        return len(self.filelabels)
    
    def __getitem__(self, ix):
        return self.tokenizer(self.files[ix].read_text()), self.filelabels[ix]
    
    def max_len(self):
        len_max = 0
        for i in range(self.__len__()):
            t = self.__getitem__(i)
            if len(t[0]) > len_max:
                len_max=len(t[0])
        
class Attention_base_model(torch.nn.Module):
    def __init__(self,dim_emb,len_seq):
        super(Attention_base_model,self).__init__()
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



class Attention_simple(torch.nn.Module):
    def __init__(self,dim_emb,len_seq):
        super(Attention_simple,self).__init__()
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


WORDS = re.compile(r"\S+")
def tokenizer(t):
    return list([x for x in re.findall(WORDS, t.lower())])

train_data = FolderText(ds.train.classes, tokenizer, load=False)
test_data = FolderText(ds.test.classes, tokenizer, load=False)

n_epoch = 1000

batch_size=20
learning_rate=10e-3


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
        
    for i,(x,y) in enumerate(test_loader):
        with torch.no_grad():
            model.eval()
            pred = model(x)
            loss = criterion(pred,y)
            # print('loss', loss ," Prédiction : ", pred, "y True : ",y)
            writer.add_scalar('Loss/validation', loss, ep)