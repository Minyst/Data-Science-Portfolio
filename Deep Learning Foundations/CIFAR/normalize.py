import torch
from model import SimpleCNN
from glob import glob
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

train_imgs = glob(os.path.join("C:/CIFAR/cifar-100-images/train", "*/*.png"))
transform = transforms.Compose([transforms.ToTensor()])

r = []
g = []
b = []

for train_img in train_imgs:
    img = Image.open(train_img).convert("RGB")
    img_to_tensor = transform(img)
    r.append(img_to_tensor[0])
    g.append(img_to_tensor[1])
    b.append(img_to_tensor[2])
    
r = np.array(r)
g = np.array(g)
b = np.array(b)

print(r.mean(), g.mean(), b.mean(), r.std(), g.std(), b.std())
