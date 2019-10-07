import torch
"""
    Yannis Karmim et Marc Treu
"""

class Highway(torch.nn.Module):

    def __init__(self, d_in):

        self.linear = torch.nn.Linear(d_in, d_in)
        self.transform = torch.nn.Linear(d_in, d_in)

        self.H = torch.nn.ReLU()
        self.T = torch.nn.Sigmoid()

    def forward(self, x):
        
        gate = self.T(self.transform(x))

        return torch.matmul(self.H(self.linear(x)),gate) + torch.matmul(1 - gate, x)