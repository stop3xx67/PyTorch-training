from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs
# import torchvision.transforms.v2 as tfs_v2 - недоступен на Stepik

# здесь объявляйте класс AddNoise

img_pil = Image.new(mode="RGB", size=(128, 128), color=(0, 128, 255))

# здесь продолжайте программу
class AddNoise(nn.Module):
    def __init__(self, volume, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.volume = volume