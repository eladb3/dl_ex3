import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

debug = False

def print_during_train(epoch, loss, acc):
    s = f'>>> Epoch {epoch}, '
    s += f"train: loss {loss['train'][-1]:.4f} acc {acc['train'][-1]:.4f}, "
    s += f"test: loss {loss['test'][-1]:.4f} acc {acc['test'][-1]:.4f}, "
    print(s)


####### NMS

def get_dimtag(dim, k, s):
    return ((dim - k) // s) + 1


def get_boxes(Htag, Wtag, k, s):
    N = Htag * Wtag
    res = torch.ones((N, 4))
    res[:, 1] = (torch.arange(N) // Wtag) * s # y1
    res[:, 0] = (torch.arange(N) %  Wtag) * s # x1
    res[:, 2] = res[:, 0] + k                 # y2
    res[:, 3] = res[:, 1] + k                 # x2
    return res

def nms(fcn_scores, iou_threshold, k =12, s = 2):
    with torch.no_grad():
        N = fcn_scores.size(0)
        H, W = fcn_scores.shape[-2:]
        boxes = get_boxes(H, W, k = k, s = s).to(device)
        res = torch.ones((N, boxes.size(0), 5)).to(device) * -1
        ls = []
        for i in range(N):
            f = fcn_scores[i, 1, :, :].flatten()
            keep = torchvision.ops.nms(boxes = boxes, scores = f, iou_threshold = iou_threshold) 
            res[i, :keep.size(0), :4] = boxes[keep, :]
            res[i, :keep.size(0),  4] = f[keep]
            ls.append(int(keep.size(0)))
    return res, ls

## FDDB

def gen_elipse(x1, y1, x2, y2):
    angle = 1.57 # pi/2 
    dx, dy = x2-x1, y2-y1
    x_center = x1 + (dx/2)
    y_center = y1 + (dy/2)
    if dx > dy: angle *= 2
    major = max(dx , dy) /2
    minor = min(dx, dy) /2
    return f'{major:.2f} {minor:.2f} {angle:.2f} {x_center:.2f} {y_center:.2f}'

def gen_fddb_out(model, name = None):
    base = "./data/EX2_data/fddb/images/"
    with open("data/EX2_data/fddb/FDDB-folds/FDDB-fold-01.txt", 'rt') as f:
        lines = f.readlines()
    p = f"fddb-test/fold-01-out.txt"
    if name is not None:
        p = f"fddb-test/{name}/fold-01-out.txt"
        if not os.path.isdir(f"fddb-test/{name}"):
               os.mkdir(f"fddb-test/{name}")
    f_write = open(p, 'wt')
    for l in tqdm(lines):
        l = l.rstrip('\n')
        if len(l) == 0: break
        out = model(f'{base}{l}.jpg')
        s = ""
        count = 0
        for i in range(int(out.size(0))):
            x1, y1, x2, y2 ,score = out[i, :]
            count += 1
            s += f'{gen_elipse(x1, y1, x2, y2)} {score:.2f}\n'
        f_write.write(f"{l}\n{count}\n{s}")
    f_write.close()
    return
            
            
    



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


# Plot

def plt_tensor(x):
    plt.imshow(TF.to_pil_image(x))

def plot_with_rec(x, locs, scale = 1, fill = False, limit = 0):
    if not isinstance(x, torch.Tensor):
        x = TF.to_tensor(x)
    H, W = x.shape[-2:]
    for i in range(locs.size(0)):
        x1, y1, x2, y2, s = locs[i, :]
        if s < limit: continue
        assert x2 > x1 and y2 > y1, print(f'index {i}')
        if scale != 1:
            x1, y1, x2, y2 = map(lambda i: int(i/scale), (x1, y1, x2, y2))
        # Display the image
        plt.imshow(TF.to_pil_image(x))
        # Add the patch to the Axes
        plt.gca().add_patch(Rectangle((x1, y1),y2 - y1, x2 - x1,linewidth=1,edgecolor='r',facecolor='r' if fill else "none", alpha = float(s/3) if fill else None))
