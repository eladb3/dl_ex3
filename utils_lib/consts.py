import torch
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
EX2_data_path = "./data/EX2_data"
VOC_data_path = "./data/VOC"