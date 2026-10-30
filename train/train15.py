import torch

# значения списков w и g в программе не менять
w = list(map(float, input().split()))
g = list(map(float, input().split()))

W1 = torch.tensor([
    [w[1], w[2], w[3]],
    [w[-3], w[-2], w[-1]]
    ])

bias1 = torch.tensor([w[0], w[4]])
W2 = torch.tensor([
    g[1], g[2]
    ])

bias2 = torch.tensor([g[0]])
t_inp = torch.rand(3) * 10

u = torch.sigmoid(torch.mv(W1, t_inp) + bias1)
y = torch.dot(W2, u) + bias2
