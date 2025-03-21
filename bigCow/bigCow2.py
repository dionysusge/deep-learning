from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class NSCHOOLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # 获取所有类别文件夹
        self.class_to_idx = {tag: class_name for class_name, tag in enumerate(self.classes)}
        self.samples = []

        # 遍历所有类别文件夹，收集样本路径和标签
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')  # 确保转换为 RGB

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__(idx + 1)

#训练集转换且增强
train_transform = transforms.Compose([
    transforms.Resize(152),       # 调整尺寸
    transforms.RandomHorizontalFlip(),   # 随机水平翻转
    transforms.RandomRotation(15),       # 随机旋转
    transforms.ToTensor(),               # 转为 Tensor
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])#像素缩放在0-1之间
])

#测试集转换且不增强
test_transform = transforms.Compose([
    transforms.Resize(152),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])#像素缩放在0-1之间
])

train_dataset = NSCHOOLDataset('../Ndataset/train',transform=train_transform)
test_dataset = NSCHOOLDataset('../Ndataset/test',transform=test_transform)

batch_size = 1

train_loader = DataLoader(train_dataset,batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size,shuffle=False)

model = models.resnet18(pretrained=True)

num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

# 评估模型
evaluate_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'Ndeepface_model.pth')