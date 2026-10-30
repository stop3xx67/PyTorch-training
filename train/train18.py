import torch
from torch.nn import MSELoss

# значения x, func, predict не менять
x = torch.arange(-3, 3, 0.1)
func = x ** 2 - 2 * torch.cos(x) - 5
predict = func + torch.empty_like(func).normal_(0, 0.5)

loss_func = MSELoss()

Q = loss_func(predict, func).item()

Q_mse = torch.mean((predict - func) ** 2).item()