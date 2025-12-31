import torch
import torch.nn as nn

def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    训练一个 epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    print(f'[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%')
    return avg_loss, accuracy

def evaluate(model, device, test_loader, criterion):
    """
    在测试集上评估模型
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f'Test Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%')
    return avg_loss, accuracy
