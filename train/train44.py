from PIL import Image
import torchvision.transforms as tfs
# import torchvision.transforms.v2 as tfs_v2 - недоступен на Stepik

img_pil = Image.new(mode="RGB", size=(128, 128), color=(0, 128, 255))

# здесь продолжайте программу
transformers = tfs.Compose([
    tfs.CenterCrop(128),
    tfs.Resize(128),
    tfs.ToTensor()
    ])

img = transformers(img_pil)