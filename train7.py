import torch

current_tm = int(input()) # переменную current_tm в программе не менять

hour = current_tm // 3600
minute = current_tm // 60 - hour * 60
second = current_tm - hour * 3600 - minute * 60

t_time = torch.tensor([current_tm, hour, minute, second], dtype=torch.int32)