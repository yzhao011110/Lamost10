# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:02:43 2024

@author: 22120
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义 CNN + Transformer 模型
class CNNTransformerModel(nn.Module):
    def __init__(self, num_classes, seq_len, input_dim, transformer_hidden_dim, transformer_layers, nhead):
        super(CNNTransformerModel, self).__init__()

        # CNN 层：提取局部特征
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=15, padding=7)  # 输出通道为 64
        self.bn1 = nn.BatchNorm1d(64)  # Batch Normalization 层
        self.pool = nn.MaxPool1d(5)

        # 第二个卷积层
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=15, padding=7)  # 输出通道为 64
        self.bn2 = nn.BatchNorm1d(64)  # Batch Normalization 层

        # 第三个卷积层
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=15, padding=7)  # 输出通道为 64
        self.bn3 = nn.BatchNorm1d(64)  # Batch Normalization 层

        # 第四个卷积层
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, padding=3)  # 输出通道为 64
        self.bn4 = nn.BatchNorm1d(64)  # Batch Normalization 层

        # Transformer 层
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=nhead,batch_first=True), num_layers=transformer_layers
        )

        # Dropout 层
        self.dropout = nn.Dropout(0.4)

        # 扩展维度的全连接层
        self.fc3 = nn.Linear(64, 1024)  # 将 Transformer 输出的维度 (64) 映射到 (1024)

        # 全连接层1（fc1），输出维度为 1024
        self.fc1 = nn.Linear(1024, 512)  # 继续保持 1024 维度

        # 全连接层2（fc2），输出类别数
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 转置数据：从 [batch_size, seq_len, features] 转为 [batch_size, features, seq_len]
        x = x.transpose(1, 2)  # (batch_size, seq_len, features) -> (batch_size, features, seq_len)

        # CNN 部分：提取局部特征
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, seq_len)
        x = self.pool(x)  # (batch_size, 64, seq_len//5)

        x = F.relu(self.bn2(self.conv2(x)))  # (batch_size, 64, seq_len//5)
        x = self.pool(x)  # (batch_size, 64, seq_len//25)

        x = F.relu(self.bn3(self.conv3(x)))  # (batch_size, 64, seq_len//25)
        x = self.pool(x)  # (batch_size, 64, seq_len//125)

        x = F.relu(self.bn4(self.conv4(x)))  # (batch_size, 64, seq_len//125)
        x = self.pool(x)  # (batch_size, 64, seq_len//625)

        # Transformer 部分：提取长期依赖
        x = x.transpose(1, 2)  # (batch_size, seq_len//625, 64) -> (batch_size, seq_len//625, feature_dim)
        x = self.transformer_encoder(x)  # (batch_size, seq_len//625, 64)

        # 对 Transformer 输出进行池化，得到 (batch_size, 64)
        x = x.mean(dim=1)

        # Dropout
        x = self.dropout(x)

        # 将 Transformer 输出的维度 (64) 映射到 (1024)
        x = F.relu(self.fc3(x))  # (batch_size, 1024)

        # 通过全连接层1（fc1）进一步处理
        x = F.relu(self.fc1(x))  # (batch_size, 1024)

        # 最终分类输出
        x = self.fc2(x)  # (batch_size, num_classes)

        return x


# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, file_paths, labels, seq_length=3700, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.seq_length = seq_length  # 数据序列长度
        self.transform = transform  # 数据增强或标准化（可选）
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 假设数据已经保存在 CSV 文件中
        data = pd.read_csv(self.file_paths[idx], skiprows=1, header=None).values  # 读取数据为 numpy 数组
        
        # 确保数据是 float32 类型
        data = np.array(data, dtype=np.float32)
        
        # 对第一列进行标准化，第二列保持原样
        data[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()
        
        # 转换为 torch Tensor
        data = torch.tensor(data, dtype=torch.float32)
        
        # 裁剪或填充数据到指定的序列长度
        current_length = data.size(0)
        if current_length > self.seq_length:
            data = data[:self.seq_length, :]
        elif current_length < self.seq_length:
            # 填充（pad）数据
            padding = self.seq_length - current_length
            data = F.pad(data, (0, 0, 0, padding))
        
        # 如果有 transform（如标准化），应用于数据
        if self.transform:
            data = self.transform(data)
        
        label = self.labels[idx]
        
        return data, label


# 获取文件路径和标签的函数
def get_file_paths_and_labels(data_dir, class_mapping):
    file_paths = []
    labels = []
    for class_name, class_idx in class_mapping.items():
        class_folder = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_folder):
            if file_name.endswith('.csv'):
                file_paths.append(os.path.join(class_folder, file_name))
                labels.append(class_idx)
    return file_paths, labels


def evaluate(model, data_loader, device, class_mapping):
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算分类报告，包括 Precision, Recall, 和 F1 Score
    report = classification_report(
        all_labels, all_preds, target_names=class_mapping.keys(), output_dict=True
    )

    # 打印分类结果
    print("Classification Report:")
    for class_name, metrics in report.items():
        if class_name in class_mapping.keys():  # 针对具体类别
            print(f"Class {class_name}: Precision = {metrics['precision']:.2f}, Recall = {metrics['recall']:.2f}, F1 = {metrics['f1-score']:.2f}")

    # 宏平均指标
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']

    print(f"\nMacro Precision: {macro_precision:.2f}")
    print(f"Macro Recall: {macro_recall:.2f}")
    print(f"Macro F1 Score: {macro_f1:.2f}")

    return macro_precision, macro_recall, macro_f1


# 生成混淆矩阵并可视化
def generate_confusion_matrix(model, data_loader, device):
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    return cm

# 可视化损失和准确率
# 可视化损失和准确率并保存为文件
def plot_metrics(losses, accuracies, val_accuracies, num_epochs):
    epochs = range(1, num_epochs + 1)

    # 绘制训练损失图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss', color='b')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 绘制训练准确率和验证准确率图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Training Accuracy', color='g')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='r')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    # 保存损失和准确率图像
    plt.savefig('training_metrics.png')
    print("Training metrics saved as 'training_metrics.png'")
    plt.close()  # 关闭当前图像


# 设置设备：如果有 GPU，则使用 GPU，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型初始化
input_dim = 2  # 输入特征维度
num_classes = 9  # 类别数
transformer_hidden_dim = 128  # Transformer 隐藏层维度
transformer_layers = 1  # Transformerencoder 数
seq_len = 3700  # 序列长度
batch_size = 128
# 初始化模型
model = CNNTransformerModel(
    num_classes=num_classes,
    seq_len=seq_len,
    input_dim=input_dim,
    transformer_hidden_dim=transformer_hidden_dim, 
    transformer_layers=transformer_layers, 
    nhead=16  # 设置 nhead 为 16
)

model.to(device)

# 打印模型结构
print(model)

# 获取训练集、验证集和测试集的数据
train_dir = './train'
val_dir = './val'
test_dir = './test'

# 类别映射
class_mapping = {'A': 0, 'B': 1, 'F': 2, 'G': 3, 'K': 4, 'M': 5, 'O': 6, 'QSO': 7, 'GALAXY': 8}

# 获取训练集、验证集和测试集的数据
train_paths, train_labels = get_file_paths_and_labels(train_dir, class_mapping)
val_paths, val_labels = get_file_paths_and_labels(val_dir, class_mapping)
test_paths, test_labels = get_file_paths_and_labels(test_dir, class_mapping)

# 数据集
train_dataset = MyDataset(file_paths=train_paths, labels=train_labels)
val_dataset = MyDataset(file_paths=val_paths, labels=val_labels)
test_dataset = MyDataset(file_paths=test_paths, labels=test_labels)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 分类问题，使用交叉熵损失

# 使用 AdamW 优化器
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # 权重衰减 1e-4

# 初始化空数组，用于存储每个 epoch 的损失、准确率和验证准确率
losses = []  # 用于存储每个 epoch 的损失
accuracies = []  # 用于存储每个 epoch 的训练准确率
val_accuracies = []  # 用于存储每个 epoch 的验证准确率

# 训练过程
num_epochs =100

# 训练过程
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算损失
        running_loss += loss.item()

        # 计算预测和标签
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算训练集准确率
    train_accuracy = accuracy_score(all_labels, all_preds) * 100

    # 在验证集上计算准确率
    model.eval()  # 切换到评估模式
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    # 计算验证集准确率
    val_accuracy = accuracy_score(val_labels, val_preds) * 100

    # 打印训练信息
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

    # 将每个 epoch 的损失、训练准确率、验证准确率加入相应的数组
    losses.append(running_loss / len(train_loader))
    accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

# 训练结束后，进行测试集评估
print("\nEvaluating on Test Set:")

# 测试集评估
test_preds = []
test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# 计算测试集指标
test_accuracy = accuracy_score(test_labels, test_preds) * 100
test_precision = precision_score(test_labels, test_preds, average='weighted')
test_recall = recall_score(test_labels, test_preds, average='weighted')
test_f1 = f1_score(test_labels, test_preds, average='weighted')

# 打印测试集指标
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test Precision: {test_precision:.2f}")
print(f"Test Recall: {test_recall:.2f}")
print(f"Test F1 Score: {test_f1:.2f}")



# 生成并绘制混淆矩阵
cm = generate_confusion_matrix(model, test_loader, device)

# 可视化混淆矩阵并保存为文件
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_mapping.keys(), yticklabels=class_mapping.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# 保存混淆矩阵图像
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.close()  # 关闭当前图像

# 训练结束后，调用可视化函数
plot_metrics(losses, accuracies, val_accuracies, num_epochs)

# 在测试集上评估 Precision、Recall 和 F1
precision, recall, f1 = evaluate(model, test_loader, device, class_mapping)
print(f"Test Precision: {precision:.2f}")
print(f"Test Recall: {recall:.2f}")
print(f"Test F1 Score: {f1:.2f}")

# 保存模型的 state_dict
torch.save(model.state_dict(), 'cnn_transformer_model.pth')
print("Model parameters saved successfully.")
