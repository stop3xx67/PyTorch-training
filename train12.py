import torch

# переменные lst и t_box_lst в программе не менять
lst = list(map(float, input().split()))
t_box_lst = torch.tensor(lst, dtype=torch.int32).view(-1, 3)

V = t_box_lst.prod(dim=1)

t_mean_vol = V.float().mean().item()
t_min_vol = V[t_mean_vol > V].min().item()
t_max_vol = V[t_mean_vol < V].max().item()