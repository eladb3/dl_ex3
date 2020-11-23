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
import torchvision.transforms.functional as TF
import random
from utils_lib.consts import device, VOC_data_path, EX2_data_path

def lua2torch(path):
    f= torchfile.load(path)
    return list(zip(torch.tensor(list(f.values()), dtype = torch.float32), [torch.tensor(1, dtype = torch.int64) for i in f.values()]))

def has_person(target):
    for ob in target['annotation']['object']:
        if ob['name'] == 'person': return True
    return False
    pass

def crop(x, c):
    x1, y1, x2, y2 = map(int, tuple(c[:4].numpy()))
    return x[:, y1:y2, x1:x2]

def mining(model, data, min_score = 0.99, max_per = 5):
    """
    model: 12Net
    data: tensor of size (3, H, W)
    """
    out = model(x = data, iou_threshold = 0.1, scales = [0.08, 0.15, 0.3]).cpu()
    out = out[out[:, -1] > min_score, :-1]
    out = out[torch.randperm(out.size(0)), :]
    res = []
    for i in range(min(len(out), max_per)):
        cr = crop(data, out[i, :])
        res.append(TF.resize(cr, (24, 24)))
    return res

def get_VOC(transform):
    download = not os.path.isdir(VOC_data_path)
    datas = []
    for t in ['train', 'trainval', 'val']:
        d = torchvision.datasets.VOCDetection(root = VOC_data_path,
                                             image_set = 'train',
                                             transform = transform,
                                             download = download)
        datas.append(d)
    data = datas[0] + datas[1] + datas[2]
    return data

def gen_noface_data(n, size):

    def gen_noface_data_internal(size, k):
        transform = transforms.Compose([
            transforms.RandomResizedCrop((size, size), scale = (0.05, 1)),
            transforms.ToTensor(),
        ])
        data = get_VOC(transform)
        n_no_persons = 14332 #pre calculated
        l = []
        for i in range(len(data)):
            if has_person(data[i][1]): continue
            l.append((data[i][0], torch.tensor(0, dtype = torch.int64)))
            if len(l) >= k: break
        return l

    res = []
    while len(res) < n:
        res.extend(gen_noface_data_internal(size, n - len(res)))
    return res[:n]

def gen_noface24_data(detector_model, step = 500, continue_ = False):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data = get_VOC(transform)
    #     l = [data[i][0] for i in range(len(data)) if not has_person(data[i][1])]
    print(f">> Start generating noface crops out of {len(data)} images") # 5717
    parts = len(data) // step + 1
    if not os.path.isdir("cache/tmp"): os.mkdir("cache/tmp")
    base = f"cache/tmp/utils.data.MyData24.noface.parts"
    n = 0
    
    start = 0
    if continue_:
        for start in range(parts):
            if not os.path.isfile(f"{base}.{start}"): break
    
    for s in range(start, parts):
        res = []
        for i in range(s * step, min((s+1) * step, len(data))):
            H, W = data[i][0].shape[-2:]
            if has_person(data[i][1]) or H < 24 or W < 24: continue
            o = mining(detector_model, data[i][0])
            res.extend(zip(o, [torch.tensor(0, dtype = torch.int64)] * len(o))) 
        torch.save(res, f"{base}.{s}")
        n += len(res)
        print(f"Scanned {(s+1) * step} images, generated {n} images")
    
    #concatenate
    res = []
    for p in range(parts):
        res.extend(torch.load(f"{base}.{p}"))
        os.remove(f"{base}.{p}")
    if len(os.listdir("cache/tmp")) == 0: os.rmdir("cache/tmp")
    return res

def split_train_test(data, test_ratio = 0.2):
    n = len(data)
    train_idxs = np.random.choice(range(n), size = int(n * test_ratio), replace = False)
    test_idxs = set(range(n)) - set(train_idxs)
    assert(len(set(train_idxs).intersection(test_idxs)) == 0)
    return [data[i] for i in train_idxs], [data[i] for i in test_idxs]

################# Faces Data

class  MyData12:
    def init(self, cache = True):
        print("Prepare 12Net data")
        self.aflw_path = "cache/utils.data.MyData12.aflw"                  
        if not cache or not os.path.isfile(self.aflw_path):
            aflw = lua2torch(f"{EX2_data_path}/aflw/aflw_12.t7")
            torch.save(aflw, self.aflw_path)
        
        self.noface_path = "cache/utils.data.MyData12.noface"
        if not cache or not os.path.isfile(self.noface_path):
            noface = gen_noface_data(len(self.aflw()), size = 12)
            torch.save(noface, self.noface_path)
        
        data = {}
        self.data_path = "cache/utils.data.MyData12.data"
        if not cache or not os.path.isfile(self.data_path):
            data['train'], data['test'] = \
                split_train_test(self.aflw() + self.noface())
            torch.save(data, self.data_path)
        print("Finished")
    def aflw(self):
        return torch.load(self.aflw_path)
    def noface(self):
        return torch.load(self.noface_path)
    def data(self):
        return torch.load(self.data_path)
    
    def DataLoader(self, batch_size, typ):
        data = self.data()[typ]
        pin_memory = torch.cuda.is_available()
        return torch.utils.data.DataLoader(data, 
                                           batch_size  = batch_size, 
                                           shuffle = True, 
                                           pin_memory = False)


class  MyData24:
    
    def init(self, detector_model, cache = True, neg_mining = True, continue_ = False):
        print("Prepare 24Net data...")
        base = "cache/utils.data.MyData24"
        self.aflw_path = f"{base}.aflw"                  
        if not cache or not os.path.isfile(self.aflw_path):
            aflw = lua2torch(f"{EX2_data_path}/aflw/aflw_24.t7")
            torch.save(aflw, self.aflw_path)
        
        n = len(self.aflw())
        if neg_mining:
            self.noface_path = f"{base}.noface"
            if not cache or not os.path.isfile(self.noface_path):
                noface = gen_noface24_data(detector_model, continue_ = continue_)
                random.shuffle(noface)
                if len(noface) > n: noface = noface[:n]
                torch.save(noface, self.noface_path)
        else:
            self.noface_path = f"{base}.noface_regular"
            if not cache or not os.path.isfile(self.noface_path):
                noface = gen_noface_data(n, 24)
                torch.save(noface, self.noface_path)

        data = {}
        self.data_path = f"{base}.data"
        if not cache or not os.path.isfile(self.data_path):
            data['train'], data['test'] = \
                split_train_test(self.aflw() + self.noface())
            torch.save(data, self.data_path)
        print("Finished")
        
    def aflw(self):
        return torch.load(self.aflw_path)
    def noface(self):
        return torch.load(self.noface_path)
    def data(self):
        return torch.load(self.data_path)
    
    def DataLoader(self, batch_size, typ):
        data = self.data()[typ]
        pin_memory = torch.cuda.is_available()
        return torch.utils.data.DataLoader(data, 
                                           batch_size  = batch_size, 
                                           shuffle = True, 
                                           pin_memory = pin_memory)


    
# constants

mData12 = MyData12()
mData24 = MyData24()
