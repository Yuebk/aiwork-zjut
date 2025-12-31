import torch
import torchvision
import torchvision.transforms as transforms
import os

def get_data_loaders(batch_size=64, data_root='./data'):
    """
    获取 CIFAR-10 数据加载器 (含数据增强，与 LeNet Advanced 保持一致)
    
    Args:
        batch_size (int): 批次大小
        data_root (str): 数据集存放根目录
        
    Returns:
        train_loader, test_loader, classes
    """
    
    # CIFAR-10 的均值和标准差
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # 1. 训练集预处理：加入数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # 随机裁剪
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    # 2. 测试集预处理：仅标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    # 3. 加载训练集
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, 
        train=True,
        download=True, 
        transform=train_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )
    
    # 4. 加载测试集
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, 
        train=False,
        download=True, 
        transform=test_transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
               
    return train_loader, test_loader, classes
