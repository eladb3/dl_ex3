import torch
from torch import nn
import torchvision
debug = False

def print_during_train(epoch, loss, acc):
    s = f'>>> Epoch {epoch}, '
    s += f"train: loss {loss['train'][-1]:.4f} acc {acc['train'][-1]:.4f}, "
    s += f"test: loss {loss['test'][-1]:.4f} acc {acc['test'][-1]:.4f}, "
    print(s)


####### NMS

def get_boxes(dim, k = 11, s = 2):
    """
    dim = H = W
    """
    dtag = ((dim - k) // s) + 1
    N = dtag**2
    res = torch.ones((N, 4))
    res[:, 1] = (torch.arange(N) // dtag) * s # y1
    res[:, 0] = (torch.arange(N) %  dtag) * s # x1
    res[:, 2] = res[:, 0] + k                 # y2
    res[:, 3] = res[:, 1] + k                 # x2
    return res

def nms(dim, fcn_scores, iou_threshold):
    with torch.no_grad():
        N = fcn_scores.size(0)
        boxes = get_boxes(dim)
        res = torch.ones((N, boxes.size(0), 5)) * -1
        for i in range(N):
            f = fcn_scores[i, 1, :, :].flatten()
            keep = torchvision.ops.nms(boxes = boxes, scores = f, iou_threshold = iou_threshold) 
            res[i, :keep.size(0), :4] = boxes[keep, :]
            res[i, :keep.size(0),  4] = f[keep]
    return res




####### Layers
class nn_Reshape(nn.Module):
    
    def __init__(self, dim = []):
        super(nn_Reshape, self).__init__()
        self.dim = list(dim)
    
    def forward(self, x):
        if debug: print(f"Dim before reshape: {x.shape}")
        if not self.dim: return x
        x = x.view([x.shape[0]] + self.dim)
        if debug: print(f"Dim after reshape: {x.shape}")
        return x
        
class nn_Unsqueeze(nn.Module):

    def __init__(self, dim):
        super(nn_Unsqueeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        if debug: print(f"Shape before unsqueeze: {x.size()}")
        x = x.unsqueeze(self.dim)
        if debug: print(f"Shape after unsqueeze: {x.size()}")
        return x

class nn_Squeeze(nn.Module):

    def __init__(self, dim):
        super(nn_Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        if debug: print(f"Shape before squeeze: {x.size()}")
        x = x.squeeze(dim = self.dim + 1) # mini batches is dim 0
        if debug: print(f"Shape after squeeze: {x.size()}")

        return x

