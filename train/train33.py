import torch
import torch.nn as nn
import torch.utils.data as data

# здесь продолжайте программу
def func(x):
    return 2 * torch.exp(-x/2) + 0.2 * torch.sin(x/10) - 5

coord_x = torch.arange(-5, 5, 0.1)

class FuncDataset(data.Dataset):
    def __init__(self):
        self.features = coord_x
        self.targets = func(coord_x)


    def __getitem__(self, index):
        return self.features[index], self.targets[index]


    def __len__(self):
        return len(self.targets)

d_train = FuncDataset()
x13, y13 = d_train[13]
total = len(d_train)
    