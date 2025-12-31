import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetAdvanced(nn.Module):
    def __init__(self):
        super(LeNetAdvanced, self).__init__()
        # 1. 卷积层 1
        # 输入: 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6) # 加入 BN 层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 2. 卷积层 2
        # 输入: 6 x 14 x 14
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16) # 加入 BN 层
        
        # 3. 全连接层
        # Flatten: 16 * 5 * 5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # Conv1 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv2 -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        
        # FC2 (Output)
        x = self.fc2(x)
        return x
