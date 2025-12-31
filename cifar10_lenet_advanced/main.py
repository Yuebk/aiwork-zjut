import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import csv
import matplotlib.pyplot as plt

from data_loader import get_data_loaders
from model import LeNetAdvanced
from train_eval import train_one_epoch, evaluate

def save_history(history, prefix):
    os.makedirs('outputs', exist_ok=True)
    csv_path = os.path.join('outputs', f"{prefix}_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        for idx, (tl, ta, vl, va) in enumerate(zip(
            history['train_loss'], history['train_acc'], history['test_loss'], history['test_acc']
        ), start=1):
            writer.writerow([idx, tl, ta, vl, va])
    print(f"Metrics saved to {csv_path}")

def plot_history(history, prefix):
    os.makedirs('outputs', exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 4))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    img_path = os.path.join('outputs', f"{prefix}_curves.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Curves saved to {img_path}")

def main():
    # 1. 配置参数 (保持与基础模型一致)
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 10
    DATA_ROOT = './data'
    
    # 2. 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. 准备数据 (含增强)
    print("Loading data (with augmentation)...")
    train_loader, test_loader, classes = get_data_loaders(BATCH_SIZE, DATA_ROOT)
    print("Data loaded.")
    
    # 4. 初始化模型 (Advanced)
    model = LeNetAdvanced().to(device)
    print(model)
    
    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 6. 训练循环
    print("Start training (Advanced Model)...")
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, epoch)
        
        # 评估
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print('-' * 60)
        
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")
    
    # 7. 保存模型与曲线
    os.makedirs('outputs', exist_ok=True)
    save_path = os.path.join('outputs', 'lenet_advanced_cifar10.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    save_history(history, prefix='lenet_advanced')
    plot_history(history, prefix='lenet_advanced')

if __name__ == '__main__':
    main()
