import torch
import torch.nn as nn

# здесь объявляйте класс модели (обязательно до тензора x)

batch_size = 4
x = torch.rand(batch_size, 32) # тензор x в программе не менять

# здесь продолжайте программу
class ThreeLayersModel(nn.Module):
    def __init__(self, num_input, num_neo1, num_neo2, num_output):
        super().__init__()
        self.layer1 = nn.Linear(num_input, num_neo1)
        self.layer2 = nn.Linear(num_neo1, num_neo2)
        self.layer3 = nn.Linear(num_neo2, num_output)


    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        x = torch.tanh(x)
        return x

model = ThreeLayersModel(32, 10, 12, 1)

model.train()
batch_size = 4
x = torch.rand(batch_size, 32)
predict = model(x)
model.eval()