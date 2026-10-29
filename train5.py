import torch

lst = list(map(float, input().split())) # список lst в программе не менять

t_mask = torch.empty(16, dtype=torch.int8)

for i in range(len(t_mask)):
    t_mask[i] = 1.0 if i % 2 != 0 else -1.0

tr = torch.tensor(lst, dtype=torch.float32)
tr *= t_mask

print(*[round(i.item(), 2) for i in tr])