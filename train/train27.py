import torch
import torch.nn as nn

# здесь объявляйте класс модели (обязательно до тензора x)

x = torch.rand(3) # тензор x в программе не менять

# здесь продолжайте программу
class TwoLayerModel(nn.Module):
    