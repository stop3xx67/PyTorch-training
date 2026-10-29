import torch
import numpy as np

a, b, n = map(float, input().split()) # переменные a, b, n в программе не менять

t_range = torch.linspace(a, b, int(n))

sigm = lambda x: 1 / (1 + np.exp(-x))

t_out = torch.tensor(sigm(t_range))
t_pred = t_out.argmax().item()