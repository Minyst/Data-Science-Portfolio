import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# 1. 데이터셋 준비 (CIFAR-10)
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet 입력 크기에 맞추기
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# 2. 사전 학습된 ResNet18 불러오기
model = models.resnet18(pretrained=True)

'''
ImageNet-1K
  - 클래스 수: 1,000개
  - 이미지 개수: 약 120만 장
  - 주제: 개, 고양이, 비행기, 자동차, 가구, 음식 등 일상 사물/동물 전반
'''

# 3. 마지막 분류기 수정 (CIFAR-10은 클래스 10개)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# 4. 일부 레이어는 동결 (Feature extractor)
for param in model.layer1.parameters():
    param.requires_grad = False

# 5. GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 6. 손실함수 & 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 학습 루프
epochs = 3
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"[Epoch {epoch+1}] loss: {running_loss/len(trainloader):.4f}")

# 8. 테스트 정확도 확인
correct, total = 0, 0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
