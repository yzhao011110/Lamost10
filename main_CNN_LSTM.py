import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import numpy as np

# 定义 CNN + LSTM 模型
class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes, seq_len, input_dim, lstm_hidden_dim, lstm_layers):
        super(CNNLSTMModel, self).__init__()
        
        # CNN 层：提取局部特征
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=20, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(20)  # 添加批归一化
        
        self.pool = nn.MaxPool1d(5)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(20)  # 添加批归一化
        
        # 第三个卷积层
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(20)  # 添加批归一化
        
        # 第四个卷积层
        self.conv4 = nn.Conv1d(in_channels=20, out_channels=15, kernel_size=6, padding=3)
        self.bn4 = nn.BatchNorm1d(15)  # 添加批归一化
        
        # LSTM 层：提取时间序列的长期依赖
        self.lstm = nn.LSTM(input_size=15, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        
        # Dropout 层
        self.dropout = nn.Dropout(0.3)
        
        # 全连接层
        self.fc1 = nn.Linear(lstm_hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # 转置数据：从 [batch_size, seq_len, features] 转为 [batch_size, features, seq_len]
        x = x.transpose(1, 2)  # (batch_size, seq_len, features) -> (batch_size, features, seq_len)

        # CNN 部分：提取局部特征
        x = F.relu(self.bn1(self.conv1(x)))  # 卷积 + 批归一化 + 激活
        x = self.pool(x)  # (batch_size, 20, seq_len//5)
        
        x = F.relu(self.bn2(self.conv2(x)))  # 卷积 + 批归一化 + 激活
        x = self.pool(x)  # (batch_size, 20, seq_len//25)
        
        x = F.relu(self.bn3(self.conv3(x)))  # 卷积 + 批归一化 + 激活
        x = self.pool(x)  # (batch_size, 20, seq_len//125)
        
        x = F.relu(self.bn4(self.conv4(x)))  # 卷积 + 批归一化 + 激活
        x = self.pool(x)  # (batch_size, 15, seq_len//625)

        # LSTM 部分：提取长期依赖
        x = x.transpose(1, 2)  # (batch_size, seq_len//625, 15) -> (batch_size, seq_len//625, feature_dim)
        x, (hn, cn) = self.lstm(x)  # x: (batch_size, seq_len//625, lstm_hidden_dim)
        
        # 取 LSTM 输出的最后一层隐藏状态
        x = hn[-1]  # (batch_size, lstm_hidden_dim)
        
        # Dropout
        x = self.dropout(x)
        
        # 全连接层
        x = F.relu(self.fc1(x))  # (batch_size, 1024)
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

# 评估函数
def evaluate(model, data_loader, device):
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

    accuracy = accuracy_score(all_labels, all_preds) * 100
    return accuracy

# 设置设备：如果有 GPU，则使用 GPU，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型初始化
input_dim = 2  # 输入特征维度（假设为 1）
num_classes = 9  # 类别数
lstm_hidden_dim = 128  # LSTM 隐藏层维度
lstm_layers = 3  # LSTM 层数
seq_len = 3700  # 序列长度
batch_size = 128

# 初始化模型
model = CNNLSTMModel(num_classes=num_classes, seq_len=seq_len, input_dim=input_dim, lstm_hidden_dim=lstm_hidden_dim, lstm_layers=lstm_layers)
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
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # 权重衰减 1e-4

# 训练过程
num_epochs = 20
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
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    # 在验证集上评估
    val_accuracy = evaluate(model, val_loader, device)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# 在测试集上评估
test_accuracy = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")
