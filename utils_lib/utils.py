import torch
from torch import nn, optim
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from utils_lib.data import mData12, mData24
from copy import deepcopy
from utils_lib.consts import device, EX2_data_path
import random
import gdown

debug = False


#### Consts
model12_path = "./cache/models.12Net.best_model"
model24_path = "./cache/models.24Net.best_model"


### General

def resize(x, scale):
    H, W = x.shape[:-2]
    return TF.resize(x, size = (int(scale * H), int(scale * W)))

def get_resized_crops(x, boxes, size):
    """
    x : (3, H, W)
    boxes : (N, 5) # last if score
    """
    N = boxes.size(0)
    res = torch.zeros((N, 3, size, size), dtype = torch.float32)
    for i in range(N):
        x1, y1, x2, y2 = map(int, boxes[i, :4])
        res[i,:, :, :] = TF.resize(x[:, y1:y2, x1:x2], size = (size, size)) 
    return res

def scale_model_nms_pipeline(x, scales, fcn_model, iou_threshold):
    boxes, scores = [], []
    if not isinstance(scales, (list, tuple)):
        scales = [scales]
    with torch.no_grad():
        x = x.unsqueeze(0)
        H, W = x.shape[-2:]
        for scale in scales:
            H_scale, W_scale = int(scale * H), int(scale * W)
            xscaled = TF.resize(x, size = (H_scale, W_scale))
            xout = fcn_model(xscaled).cpu()
            Hpost, Wpost = xout.shape[-2:] 
            boxes.append(get_boxes(Hpost, Wpost, k = 12, s = 2) / scale)
            scores.append(xout[0, 1, :, :].flatten())
            assert(len(scores[-1]) == len(boxes[-1])), f"{len(scores[-1])} != {len(boxes[-1])}, {Hpost}, {Wpost}"
        
        boxes = torch.cat(boxes, 0).to(device)
        scores = torch.cat(scores, 0).to(device)
        res = torch.ones((boxes.size(0), 5)) * -1
        keep = torchvision.ops.nms(boxes = boxes, scores = scores, iou_threshold = iou_threshold).cpu()
        res[:keep.size(0), :4] = boxes[keep, :]
        res[:keep.size(0),  4] = scores[keep]
    return res[ :keep.size(0), :]

def get_random_fddb_img():
    base = f"{EX2_data_path}/fddb/images"
    path = base
    while len([p for p in os.listdir(path) if p.endswith("jpg")]) == 0:
        dirs = [d for d in os.listdir(path) if d[0] != '.' and os.path.isdir(f"{path}/{d}")]
        path = f"{path}/{dirs[random.randint(0, len(dirs)-1)]}"
    imgs = [p for p in os.listdir(path) if p.endswith("jpg")]
    return f"{path}/{imgs[random.randint(0, len(imgs)-1)]}"

def get_EX2_data():
    url = 'https://drive.google.com/u/0/uc?id=1_36ltHui4gcODBpSxgt-iPgSk7q-E2Gn'
    if not os.path.isfile("./data/EX2_data.zip"):
        output = f'{EX2_data_path}.zip'
        gdown.download(url, output, quiet=False)
    if not os.path.isdir(EX2_data_path):
        os.system(f"unzip {EX2_data_path}.zip -d ./data")
        # fix arguments
        with open(f"{EX2_data_path}/fddb/evaluation/runEvaluate.pl", 'rt') as f:
            s = f.read().replace("/home/wolf/adampolyak", EX2_data_path).replace("my $detFormat = 1;", "my $detFormat = 0;")
        with open(f"{EX2_data_path}/fddb/evaluation/runEvaluate.pl", 'wt') as f:
            f.write(s)
        
#### Prints

def print_during_train(epoch, loss, acc):
    s = f'>>> Epoch {epoch}, '
    s += f"train: loss {loss['train'][-1]:.4f} acc {acc['train'][-1]:.4f}, "
    s += f"test: loss {loss['test'][-1]:.4f} acc {acc['test'][-1]:.4f}, "
    print(s)
    
def plt_loss_acc(loss, acc):
    plt.figure(figsize=(20,5))
    plt.subplot(1,2,1)
    plt.title("Accuracy")
    plt.plot(acc['train'], label = 'train')
    plt.plot(acc['test'], label = 'test')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.title("Loss")
    plt.plot(loss['train'], label = 'train')
    plt.plot(loss['test'], label = 'test')
    plt.legend()

    plt.show()

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

def gen_ellipse(x1, y1, x2, y2):
    angle = 1.57 # pi/2 
    dx, dy = x2-x1, y2-y1
    x_center = x1 + (dx/2)
    y_center = y1 + (dy/2)
    if dx > dy: angle *= 2
    major = max(dx , dy) /2
    minor = min(dx, dy) /2
    return f'{major:.2f} {minor:.2f} {angle:.2f} {x_center:.2f} {y_center:.2f}'

def gen_fddb_out(model, ellipse = True, name = None):
    base = f"{EX2_data_path}/fddb/images/"
    with open(f"{EX2_data_path}/fddb/FDDB-folds/FDDB-fold-01.txt", 'rt') as f:
        lines = f.readlines()
    p = f"./fddb-test/fold-01-out.txt"
    if name is not None:
        p = f"./fddb-test/{name}/fold-01-out.txt"
        if not os.path.isdir(f"./fddb-test/{name}"):
               os.makedirs(f"./fddb-test/{name}")
    f_write = open(p, 'wt')
    for l in lines:
        l = l.rstrip('\n')
        if len(l) == 0: break
        out = model(f'{base}{l}.jpg')
        s = ""
        count = 0
        for i in range(int(out.size(0))):
            x1, y1, x2, y2 ,score = out[i, :]
            count += 1
            encode = gen_ellipse(x1, y1, x2, y2) if ellipse else f"{int(x1)} {int(y1)} {int(x2 - x1)} {int(y2 - y1)}"
            s += f'{encode} {score:.2f}\n'
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

def plot_with_rec(x, locs, scale = 1, fill = False, n_max = float('Inf'), limit = 0, sort= True):
    if not isinstance(x, torch.Tensor):
        x = TF.to_tensor(x)
    if sort:
        locs = torch.stack(sorted(list(locs), key=lambda x: -x[-1]))
    H, W = x.shape[-2:]
    # Display the image
    plt.imshow(TF.to_pil_image(x))

    for i in range(min(n_max, locs.size(0))):
        x1, y1, x2, y2, s = locs[i, :]
        if s < limit: continue
        assert x2 > x1 and y2 > y1, print(f'index {i}')
        if scale != 1:
            x1, y1, x2, y2 = map(lambda i: int(i/scale), (x1, y1, x2, y2))
        # Add the patch to the Axes
        plt.gca().add_patch(Rectangle((x1, y1),y2 - y1, x2 - x1,linewidth=1,edgecolor='r',facecolor='r' if fill else "none", alpha = float(s/3) if fill else None))

        
## Train

def train_12Net(Net12, Epochs, save = False):
    train_Net(Net12, Epochs, mData12, model12_path if save else None)

def train_24Net(Net24, Epochs, save = False, batch_size = 128):
    train_Net(Net24, Epochs, mData24, model24_path if save else None, add_softmax = True, print_each = 10, batch_size = 128)

def train_Net(Net, Epochs, mData, path = None, add_softmax = False, print_each = 100, batch_size = 256, early_stop = 10):
    Net.to(device)
    opt = optim.Adam(Net.parameters(), lr=  1e-4)
    loss_fn = nn.CrossEntropyLoss()
    E = Epochs
    l_loss, l_acc = {'train':[], 'test':[]}, {'train':[], 'test':[]}
    best_model, best_acc = None ,0
    dl = mData.DataLoader(typ = 'train', batch_size = batch_size)
    dl_test = mData.DataLoader(typ = 'test', batch_size = 256)
    n_train, n_test = len(dl.dataset), len(dl_test.dataset)
    early_stop_state = 0
    for epoch in range(E):
        epoch_loss, epoch_acc = 0, 0
        # train epoch
        for x, y in dl:
            x, y = x.to(device), y.view(-1).type(torch.int64).to(device)
            y_pred = Net(x)
            loss = loss_fn(y_pred, y)
            opt.zero_grad() ; loss.backward() ; opt.step()
            with torch.no_grad():
                c_acc = (y_pred.cpu().argmax(axis=1) == y.cpu()).type(torch.float32).sum()
                epoch_acc += c_acc
            epoch_loss += loss * len(x) 
#             print(list(map(float, (loss, c_acc, sum(y == 1), sum(y == 0)))))
        l_loss['train'].append(epoch_loss / n_train)
        l_acc['train'].append(epoch_acc / n_train)
        epoch_loss, epoch_acc = 0, 0
        # test
        for x,y in dl_test:
            x, y = x.to(device), y.view(-1).type(torch.int64).to(device)
            with torch.no_grad():
                y_pred = Net(x)
                loss = loss_fn(y_pred, y)
                epoch_acc += (y_pred.cpu().argmax(axis=1) == y.cpu()).type(torch.float32).sum()
            epoch_loss += loss * len(x)
        # update best model
        l_loss['test'].append(epoch_loss / n_test)
        l_acc['test'].append(epoch_acc / n_test)
        if l_acc['test'][-1] > best_acc:
            best_acc = l_acc['test'][-1]
            best_model_train_acc = l_acc['train'][-1]
            best_model = deepcopy(Net)
            early_stop_state = 0
        else: 
            early_stop_state += 1
        if early_stop_state > early_stop:
            print(f"Test accuracy has not improved in {early_stop_state} epochs, stop train")
            break
        if epoch % print_each == 0: 
            print_during_train(epoch,l_loss, l_acc)
    print(f"Best model test accuracy: {best_acc}, train accuracy: {best_model_train_acc}")
    if path is not None:
        if add_softmax: best_model = nn.Sequential(best_model, nn.Softmax(dim = 1))
        torch.save(best_model.cpu(), path)
        print(f">>> model saved to {path}")
    plt_loss_acc(l_loss, l_acc)
