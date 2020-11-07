import torchfile
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop, Compose
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import os
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def lua2torch(path):
    f= torchfile.load(path)
    return list(zip(torch.tensor(list(f.values()), dtype = torch.float32), [torch.tensor(1, dtype = torch.int64) for i in f.values()]))

def gen_noface_data(n, size):
    transform = transforms.Compose([
        transforms.RandomCrop((size, size)),
        transforms.ToTensor(),
    ])

    def gen_noface_data_internal(size):
        download = not os.path.isdir("data/VOC")
        data = torchvision.datasets.VOCDetection(root = 'data/VOC',
                                             image_set = 'train',
                                             transform = transform,
                                             download = download)
        l = [(data[i][0], torch.tensor(0, dtype = torch.int64)) for i in range(len(data)) if not has_person(data[i][1])]
        return l

    res = []
    while len(res) < n:
        res.extend(gen_noface_data_internal(size))
    return res[:n]

def has_person(target):
    for ob in target['annotation']['object']:
        if ob['name'] == 'person': return True
    return False
    pass

def split_train_test(data, test_ratio = 0.2):
    n = len(data)
    train_idxs = np.random.choice(range(n), size = int(n * test_ratio), replace = False)
    test_idxs = set(range(n)) - set(train_idxs)
    return [data[i] for i in train_idxs], [data[i] for i in test_idxs]

################# Faces Data

class  MyData:
    def __init__(self, cache = True):
        aflw_path = "cache/utils.data.MyData.aflw"
        
        if cache and os.path.isfile(aflw_path):
            self.aflw = torch.load(aflw_path)
        else:
            self.aflw = {12: lua2torch("data/EX2_data/aflw/aflw_12.t7"),
                        24: lua2torch("data/EX2_data/aflw/aflw_12.t7")}
            torch.save(self.aflw, aflw_path)
        
        noface_path = "cache/utils.data.MyData.noface"
        if cache and os.path.isfile(noface_path):
            self.noface = torch.load(noface_path)
        else:
            self.noface = {}
            for size in [12, 24]:
                self.noface[size] = gen_noface_data(len(self.aflw[size]), size)
            torch.save(self.noface, noface_path)
        self.data = {12:{}, 24:{}}
        for size in (12, 24):
            self.data[size]['train'], self.data[size]['test'] = split_train_test(self.aflw[size] + self.noface[size])

    def DataLoader(self, size, batch_size, typ):
        data = self.data[size][typ]
        pin_memory = torch.cuda.is_available()
        return torch.utils.data.DataLoader(data, 
                                           batch_size  = batch_size, 
                                           shuffle = True, 
                                           pin_memory = pin_memory)