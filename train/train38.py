import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


# здесь продолжайте программу
class Data(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x
        self.targets = _global_var_target.unsqueeze(1)
        self.length = self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.length


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.out(x)
        return x

model = Model()
model.train()

dataset = Data()
train_data = data.DataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle=True
    )

optimizer = optim.RMSprop(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(10):
    for X, y in train_data:
        optimizer.zero_grad()
        predictrion = model(X)
        loss = loss_fn(predictrion, y)
        loss.backward()
        optimizer.step()

model.eval()
predictions = model(dataset.data)
Q = loss_fn(predictions, dataset.targets).item()