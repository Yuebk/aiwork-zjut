import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVGG(nn.Module):
    def __init__(self):
        super(SimpleVGG, self).__init__()
        
        # 定义 VGG 风格的卷积块
        # Block 1: [Conv3x3 -> BN -> ReLU] x 2 -> MaxPool
        # Input: 3 x 32 x 32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32 x 16 x 16
        )
        
        # Block 2: [Conv3x3 -> BN -> ReLU] x 2 -> MaxPool
        # Input: 32 x 16 x 16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64 x 8 x 8
        )
        
        # Block 3: [Conv3x3 -> BN -> ReLU] x 2 -> MaxPool
        # Input: 64 x 8 x 8
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128 x 4 x 4
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x
