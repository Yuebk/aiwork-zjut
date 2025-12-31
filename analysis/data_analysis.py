import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
REPORT_IMG_DIR = os.path.join(ROOT_DIR, 'reports', 'img')
DATA_ROOT = os.path.join(ROOT_DIR, 'data')

os.makedirs(REPORT_IMG_DIR, exist_ok=True)

# CIFAR-10 官方统计（用于对比）
STATS = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def load_cifar(train=True):
    transform = transforms.ToTensor()
    ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=train, download=True, transform=transform)
    return ds


def compute_channel_stats(dataset):
    # 计算通道均值与标准差（使用 ToTensor 后范围 [0,1]）
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    n = 0
    mean = torch.zeros(3)
    M2 = torch.zeros(3)
    for imgs, _ in loader:
        b = imgs.size(0)
        n_batch = b * imgs.size(2) * imgs.size(3)
        # 展平到像素维度
        x = imgs.permute(1, 0, 2, 3).contiguous().view(3, -1)
        batch_mean = x.mean(dim=1)
        batch_var = x.var(dim=1, unbiased=False)
        # Welford 合并
        delta = batch_mean - mean
        tot = n + n_batch
        mean += delta * (n_batch / tot)
        M2 += batch_var * n_batch + delta**2 * (n * n_batch / tot)
        n = tot
    std = torch.sqrt(M2 / n)
    return mean.tolist(), std.tolist()


def plot_class_distribution(train_ds, val_indices=None, test_ds=None):
    # 统计训练/验证/测试各类别数量
    train_counts = [0] * 10
    for _, y in train_ds:
        train_counts[y] += 1

    val_counts = [0] * 10
    if val_indices is not None:
        for idx in val_indices:
            _, y = train_ds[idx]
            val_counts[y] += 1

    test_counts = [0] * 10
    if test_ds is not None:
        for _, y in test_ds:
            test_counts[y] += 1

    # 绘制柱状图
    x = np.arange(len(CLASSES))
    width = 0.25
    plt.figure(figsize=(12, 5))
    plt.bar(x - width, train_counts, width, label='Train')
    if val_indices is not None:
        plt.bar(x, val_counts, width, label='Val')
    if test_ds is not None:
        plt.bar(x + width, test_counts, width, label='Test')
    plt.xticks(x, CLASSES, rotation=30)
    plt.ylabel('Count')
    plt.title('CIFAR-10 Class Distribution')
    plt.legend()
    out_path = os.path.join(REPORT_IMG_DIR, 'class_distribution.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'[Saved] {out_path}')


def plot_channel_stats(train_mean_std, test_mean_std):
    (t_mean, t_std) = train_mean_std
    (v_mean, v_std) = STATS
    (te_mean, te_std) = test_mean_std

    # 绘制对比图：通道均值与标准差
    labels = ['R', 'G', 'B']
    x = np.arange(3)
    width = 0.25
    plt.figure(figsize=(10, 4))
    # 均值
    plt.subplot(1, 2, 1)
    plt.bar(x - width, t_mean, width, label='Train Mean (computed)')
    plt.bar(x, v_mean, width, label='Official Mean')
    plt.bar(x + width, te_mean, width, label='Test Mean (computed)')
    plt.xticks(x, labels)
    plt.title('Channel Mean')
    plt.legend()
    # 标准差
    plt.subplot(1, 2, 2)
    plt.bar(x - width, t_std, width, label='Train Std (computed)')
    plt.bar(x, v_std, width, label='Official Std')
    plt.bar(x + width, te_std, width, label='Test Std (computed)')
    plt.xticks(x, labels)
    plt.title('Channel Std')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(REPORT_IMG_DIR, 'channel_stats.png')
    plt.savefig(out_path)
    plt.close()
    print(f'[Saved] {out_path}')


def main():
    # 加载数据
    train_ds = load_cifar(train=True)
    test_ds = load_cifar(train=False)

    # 计算通道统计
    train_mean, train_std = compute_channel_stats(train_ds)
    test_mean, test_std = compute_channel_stats(test_ds)
    plot_channel_stats((train_mean, train_std), (test_mean, test_std))

    # 划分验证集 10% （推荐方案，报告证明比例）
    val_ratio = 0.1
    n_train = len(train_ds)
    val_size = int(n_train * val_ratio)
    val_indices = np.random.RandomState(42).choice(n_train, size=val_size, replace=False)

    # 绘制类别分布（Train/Val/Test）
    plot_class_distribution(train_ds, val_indices=val_indices, test_ds=test_ds)

    print('Done. Images saved under reports/img.')


if __name__ == '__main__':
    main()
