import torch
from SimpleCNN import SimpleCNN
from glob import glob
import os
from PIL import Image
import torchvision.transforms as transforms

model = SimpleCNN().cuda()
weight = torch.load("C:/MNIST/best_model.pth", map_location="cuda", weights_only=False)
model.load_state_dict(weight, strict=True)
test_imgs = glob(os.path.join("C:/MNIST/test", "*.png"))
transform = transforms.Compose([transforms.ToTensor()])

model.eval()
for test_img in test_imgs:
    test_img = Image.open(test_img).convert("L")
    test_img_tensor = transform(test_img).unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(test_img_tensor)
        _, pred = torch.max(outputs,1)
        predicted = predicted.item()
        print(f'{test_img} -> Predicted Label: {pred}')

# ========================================================================        

import torch
from SimpleCNN import SimpleCNN
from glob import glob
import os
from PIL import Image
import torchvision.transforms as transforms

model = SimpleCNN().cuda()
weight = torch.load("C:/MNIST/best_model.pth", map_location="cuda", weights_only=False)
model.load_state_dict(weight, strict=True)
transform = transforms.Compose([transforms.ToTensor()])
test_imgs = glob(os.path.join("C:/MNIST/test", "*.png"))

model.eval()
for test_img in test_imgs:
    test_img = Image.open(test_img).convert("L")
    test_img_tensor = transform(test_img).unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(test_img_tensor)
        pred = outputs.argmax(dim=1).item()  
        print(f"{test_img} -> Predicted Label: {pred}")

