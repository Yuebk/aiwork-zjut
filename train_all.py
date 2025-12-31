import os
import time
import csv
import importlib.util
import multiprocessing as mp
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# cuDNN 自动调优（固定尺寸卷积场景下可提升性能）
torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 使用 importlib 按文件路径加载，避免同名模块互相覆盖
MODEL_CONFIGS = {
    'lenet': {
        'model_path': os.path.join(BASE_DIR, 'cifar10_lenet', 'model.py'),
        'class_name': 'LeNet',
        'use_augmentation': False,
    },
    'lenet_advanced': {
        'model_path': os.path.join(BASE_DIR, 'cifar10_lenet_advanced', 'model.py'),
        'class_name': 'LeNetAdvanced',
        'use_augmentation': True,
    },
    'simple_vgg': {
        'model_path': os.path.join(BASE_DIR, 'cifar10_vgg_simple', 'model.py'),
        'class_name': 'SimpleVGG',
        'use_augmentation': True,
    },
}

def load_model_class(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("dyn_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return getattr(module, class_name)


def get_data_loaders(batch_size: int, data_root: str, use_augmentation: bool) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if use_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    # 在多进程并行场景下，Windows 共享内存容易不足；默认降低 worker 数
    default_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 2)
    num_workers = int(os.environ.get('NUM_WORKERS_PER_PROC', default_workers))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


def save_history(history, prefix):
    os.makedirs('outputs_all', exist_ok=True)
    csv_path = os.path.join('outputs_all', f"{prefix}_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        for idx, (tl, ta, vl, va) in enumerate(zip(
            history['train_loss'], history['train_acc'], history['test_loss'], history['test_acc']
        ), start=1):
            writer.writerow([idx, tl, ta, vl, va])
    print(f"Metrics saved to {csv_path}")


def plot_history(history, prefix):
    os.makedirs('outputs_all', exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    img_path = os.path.join('outputs_all', f"{prefix}_curves.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Curves saved to {img_path}")


def train_and_eval(model, device, train_loader, test_loader, criterion, optimizer, epochs, prefix):
    scaler = GradScaler(enabled=device.type == 'cuda')
    history = {k: [] for k in ['train_loss', 'train_acc', 'test_loss', 'test_acc']}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(enabled=device.type == 'cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast(enabled=device.type == 'cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        test_loss = val_loss / len(test_loader)
        test_acc = 100 * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"[{prefix}] Epoch {epoch}: Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | Test Loss {test_loss:.4f} Acc {test_acc:.2f}%")

    os.makedirs('outputs_all', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('outputs_all', f"{prefix}.pth"))
    save_history(history, prefix)
    plot_history(history, prefix)


def worker(model_key: str, batch_size: int, lr: float, epochs: int, data_root: str):
    cfg = MODEL_CONFIGS[model_key]
    ModelClass = load_model_class(cfg['model_path'], cfg['class_name'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{model_key}] Using device: {device}")

    train_loader, test_loader = get_data_loaders(batch_size, data_root, use_augmentation=cfg['use_augmentation'])
    model = ModelClass().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_and_eval(model, device, train_loader, test_loader, criterion, optimizer, epochs, prefix=model_key)


def main():
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 64))
    LEARNING_RATE = 0.001
    EPOCHS = 10
    DATA_ROOT = './data'

    mp.set_start_method('spawn', force=True)

    jobs = []
    for key in MODEL_CONFIGS.keys():
        p = mp.Process(target=worker, args=(key, BATCH_SIZE, LEARNING_RATE, EPOCHS, DATA_ROOT))
        p.start()
        jobs.append(p)

    for p in jobs:
        p.join()

    print("All trainings finished (parallel).")


if __name__ == '__main__':
    main()
