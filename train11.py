import torch

# переменные lst и t_rect_lst в программе не менять
lst = list(map(float, input().split()))
t_rect_lst = torch.tensor(lst, dtype=torch.int32).view(-1, 2)

S = t_rect_lst[None, 0] * t_rect_lst[None, 1]
P = (t_rect_lst[None, 0] + t_rect_lst[None, 1]) * 2

t_sum_q = S.sum().item()
t_min_per = P[S.argmin()].item()
t_max_per = P[S.argmax()].item()