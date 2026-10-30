import torch

g = float(input()) # значение g в программе не менять
d = float(input()) # значение d в программе не менять

t = torch.tensor(d, requires_grad=True)
v = (g * t) / -2 + g * torch.exp(-t) + 1.5
v.backward()