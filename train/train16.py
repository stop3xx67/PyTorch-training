import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

targets = torch.rand(1000, device=device)
predict = torch.rand(1000, device=device)

Q = torch.mean((predict - targets).pow(2)).cpu()