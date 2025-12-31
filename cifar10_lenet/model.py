import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1. 卷积层 1
        # 输入: 3 x 32 x 32
        # 卷积: 3 -> 6, kernel 5x5 => 6 x 28 x 28
        # 池化: 2x2 => 6 x 14 x 14
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 2. 卷积层 2
        # 输入: 6 x 14 x 14
        # 卷积: 6 -> 16, kernel 5x5 => 16 x 10 x 10
        # 池化: 2x2 => 16 x 5 x 5
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # 3. 全连接层
        # Flatten: 16 * 5 * 5 = 400
        # FC1: 400 -> 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        
        # FC2: 120 -> 10 (输出层)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        
        # FC2 (Output)
        x = self.fc2(x)
        return x
