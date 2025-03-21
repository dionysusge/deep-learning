from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整大小
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
])

# 加载训练数据集
train_dataset = datasets.CelebA(root="../data", split="train", download=False, transform=transform)

# 创建训练 DataLoader
train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True)

# 加载测试数据集
test_dataset = datasets.CelebA(root="../data", split="test", download=False, transform=transform)

# 创建测试数据加载器
test_iter = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建简单 CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=40):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 128 * 128, num_classes)  # 展平后全连接

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# 初始化模型
model = SimpleCNN(num_classes=40).to(device)

# 选择损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 适用于二元分类
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# 训练模型
num_epochs = 10
train_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_iter):
        images = images.to(device)
        labels = labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_iter)}], Loss: {running_loss / 100:.4f}')
            train_losses.append(running_loss / 100)
            running_loss = 0.0

    print(f"epoch {epoch + 1}: Loss: {running_loss / 100:.4f}")

print('Training finished.')

# 测试模型
test_accuracies = []
for epoch in range(num_epochs):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_iter:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            # 计算预测结果
            predicted = torch.sigmoid(outputs) > 0.5
            # 计算正确预测的数量
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    test_accuracies.append(accuracy)
    print(f'Epoch {epoch + 1}: Test Accuracy: {accuracy * 100:.2f}%')

# 绘制训练损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Steps (per 100)')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# 绘制测试准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()