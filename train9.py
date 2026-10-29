import torch

lst = list(map(int, input().split()))

targets = torch.tensor(lst, dtype=torch.int64)
t_onehot = torch.eye(targets.max()+1)[targets]

t_bag = t_onehot.sum(dim=1)

pred = torch.argmax(t_bag).item()