import torch
from torch import nn
debug = True

def print_during_train(epoch, loss, acc):
    s = f'>>> Epoch {epoch}, '
    s += f"train: loss {loss['train'][-1]:.2f} acc {acc['train'][-1]:.2f}, "
    s += f"test: loss {loss['test'][-1]:.2f} acc {acc['test'][-1]:.2f}, "
    print(s)
    
class nn_Reshape(nn.Module):
    
    def __init__(self, dim = []):
        super(nn_Reshape, self).__init__()
        self.dim = list(dim)
    
    def forward(self, x):
#         if debug: print(f"Dim before reshape: {x.shape}")
#         if not self.dim: return x
        x = x.view([x.shape[0]] + self.dim)
#         if debug: print(f"Dim after reshape: {x.shape}")
        return x
        
class nn_Unsqueeze(nn.Module):

    def __init__(self, dim):
        super(nn_Unsqueeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x = x.unsqueeze(self.dim)
        return x
