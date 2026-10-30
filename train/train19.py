import torch

d = float(input()) # значение d в программе не менять

s = 3 * d ** 2 + 5 * d - 2

s.backward()

t = torch.tensor(d.grad)