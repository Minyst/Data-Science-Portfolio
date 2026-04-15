import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from glob import glob
import shutil
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SimpleCNN
import torch.optim as optim

train_root = "C:/CIFAR/cifar-100-images/train"
val_root = "C:/CIFAR/cifar-100-images/val"
train_dir = sorted(os.listdir(train_root))

dic = {}
for idx, name in enumerate(train_dir):
    dic[name] = idx

class CIFARDataset(Dataset):
    def __init__(self, root_dir):
        self.transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4865, 0.4409),
                                            (0.2673, 0.2564, 0.2762))]) # RGB의 평균과 표준편차
        self.samples = []
        paths = sorted(glob(os.path.join(root_dir, "*/*.png")))
        for path in paths:
            path = path.replace("\\","/")
            label = dic[path.split("/")[-2]]
            self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_to_tensor = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return img_to_tensor, label
    
train_ds = CIFARDataset(train_root)
val_ds = CIFARDataset(val_root)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

model = SimpleCNN().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # gradient descent(경사하강법)를 수행하는 역할을 하는 객체
num_epochs = 10
save_init = "C:/CIFAR"

best_accuracy = 0.0  
best_epoch = 0

for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        
        #Forward(순전파) 
        optimizer.zero_grad() # 기울기 초기화
        outputs = model(inputs) 

        # crossentropy로 loss 구하기 OneHot생각하기
        loss = criterion(outputs, labels)  

        # Backward(역전파) 
        # 경사하강법(Gradient Descent)
        # learing rate와 편미분을 이용해서 가중치 조정량을 결정한다. 
        # ∂J/∂w 계산 (gradient 구하기)
        # loss.backward()는 loss 값을 바꾸는 게 아니라
        # 그 loss를 이용해서 모델 파라미터(가중치)의 gradient(기울기)를 계산하고 수정한다.
        loss.backward() 

        # 가중치를 업데이트한다. # w := w - η * gradient  (가중치 업데이트)
        # gradient → 매 배치마다 초기화 후 새로 계산 (휘발성)
        # weight → 계속 누적 업데이트 (지속성)
        # 새로운 gradient가 나올 때마다 weight가 조금씩 변하며, 점점 loss가 작은 방향으로 이동
        optimizer.step() 
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs,1)
        total += labels.size(0) # labels.size(0)는 현재 배치 크기 / 최종 누적된 total은 전체 데이터셋 크기
        correct += (predicted == labels).sum().item()

    # len(train_loader) = 배치 개수 = 전체 샘플 수 ÷ 배치 크기
    epoch_loss = running_loss / len(train_loader) 
    # total = 전체 샘플수 = 전체 데이터셋 크기
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    model.eval()
    with torch.no_grad():  # 기울기 계산을 하지 않음
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Test Accuracy after Epoch {epoch}: {test_accuracy:.2f}%")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            # 최고 성능 나올 때만 저장
            # 파일은 항상 best_model.pth 하나만 유지
            # 이후 로드할 때도 파일명 고정이라 편함
            save_path = os.path.join(save_init, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at Epoch {best_epoch} with Accuracy: {best_accuracy:.2f}%")

print(f"Best Epoch {best_epoch}, Best Accuracy: {best_accuracy:.2f}%")