import torch
from model import SimpleCNN
from glob import glob
import os
from PIL import Image
import torchvision.transforms as transforms

model = SimpleCNN().cuda()
weight = torch.load("C:/CIFAR/best_model.pth", map_location="cuda", weights_only=False)
model.load_state_dict(weight)
test_imgs = glob(os.path.join("C:/CIFAR/cifar-100-images/val", "*/*.png"))
transform = transforms.Compose([transforms.ToTensor()])

model.eval()
for test_img in test_imgs:
    test_img = Image.open(test_img).convert("RGB")
    test_img_tensor = transform(test_img).unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(test_img_tensor)
        _, pred = torch.max(outputs,0) # 질문
        print(f'{test_img} -> Predicted Label: {pred.item()}')