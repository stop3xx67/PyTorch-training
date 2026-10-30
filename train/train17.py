import torch

# все эти переменные в программе не менять, только тензоры разместить на device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W = torch.rand(3, device=device) * 10 - 5
bias = torch.rand(1, device=device) * 100 - 50

batch_size = 8 # размер мини-батча
X = torch.empty(batch_size, 3, device=device).normal_(mean=1.0, std=4.0)

predict = torch.sigmoid(torch.matmul(X, W) + bias)