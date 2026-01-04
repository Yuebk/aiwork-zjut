#C:\Users\娄苏萌\AppData\Local\Programs\Python\Python313\python.exe d:/project/aiwork-zjut/demo.py --model lenet
#C:\Users\娄苏萌\AppData\Local\Programs\Python\Python313\python.exe d:/project/aiwork-zjut/demo.py --model lenet_advanced
#C:\Users\娄苏萌\AppData\Local\Programs\Python\Python313\python.exe d:/project/aiwork-zjut/demo.py --model simple_vgg
import os
import argparse
import importlib.util
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import random

# 基础常量
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REPORT_IMG_DIR = os.path.join(BASE_DIR, 'reports', 'img', 'demo_outputs')
REPORT_DIR = os.path.join(BASE_DIR, 'reports')
DATA_ROOT = os.path.join(BASE_DIR, 'data')
os.makedirs(REPORT_IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

STATS = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

MODEL_CONFIGS = {
    'lenet': {
        'model_path': os.path.join(BASE_DIR, 'cifar10_lenet', 'model.py'),
        'class_name': 'LeNet',
        'weight_patterns': [
            os.path.join(BASE_DIR, 'outputs_all', 'lenet.pth'),
            os.path.join(BASE_DIR, 'cifar10_lenet', 'outputs', 'lenet_cifar10.pth'),
            os.path.join(BASE_DIR, 'outputs', 'lenet_cifar10.pth'),
            os.path.join(BASE_DIR, 'output', 'lenet_cifar10.pth'),
        ],
    },
    'lenet_advanced': {
        'model_path': os.path.join(BASE_DIR, 'cifar10_lenet_advanced', 'model.py'),
        'class_name': 'LeNetAdvanced',
        'weight_patterns': [
            os.path.join(BASE_DIR, 'outputs_all', 'lenet_advanced.pth'),
            os.path.join(BASE_DIR, 'cifar10_lenet_advanced', 'outputs', 'lenet_advanced_cifar10.pth'),
            os.path.join(BASE_DIR, 'outputs', 'lenet_advanced_cifar10.pth'),
            os.path.join(BASE_DIR, 'output', 'lenet_advanced_cifar10.pth'),
        ],
    },
    'simple_vgg': {
        'model_path': os.path.join(BASE_DIR, 'cifar10_vgg_simple', 'model.py'),
        'class_name': 'SimpleVGG',
        'weight_patterns': [
            os.path.join(BASE_DIR, 'outputs_all', 'simple_vgg.pth'),
            os.path.join(BASE_DIR, 'cifar10_vgg_simple', 'outputs', 'simple_vgg_cifar10.pth'),
            os.path.join(BASE_DIR, 'outputs', 'simple_vgg_cifar10.pth'),
            os.path.join(BASE_DIR, 'output', 'simple_vgg_cifar10.pth'),
        ],
    },
}


def load_model_class(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("dyn_model_demo", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return getattr(module, class_name)


def find_existing(path_list: List[str]) -> str:
    for p in path_list:
        if os.path.exists(p):
            return p
    return ''


def get_device(explicit: str = None) -> torch.device:
    if explicit:
        if explicit.lower() == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_preprocess_for_external():
    # 已不使用外部图片，仅保留用于兼容；若未来需要，可启用
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(*STATS),
    ])


def classify_tensor(model: nn.Module, device: torch.device, tensor: torch.Tensor, topk: int = 3):
    model.eval()
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = F.softmax(logits, dim=1)
        top_probs, top_idx = probs.topk(k=topk, dim=1)
    return top_probs.squeeze(0).cpu().tolist(), [CLASSES[i] for i in top_idx.squeeze(0).cpu().tolist()]


def annotate_image(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    # 半透明背景条
    pad = 4
    tw, th = draw.textlength(text, font=font), 16
    bg = Image.new('RGBA', (img.width, th + pad * 2), (0, 0, 0, 128))
    img = img.convert('RGBA')
    img.alpha_composite(bg, (0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), text, fill=(255, 255, 255, 255), font=font)
    return img.convert('RGB')

def run_on_test_samples(model_key: str, weights: str, device: torch.device, count: int):
    cfg = MODEL_CONFIGS[model_key]
    ModelClass = load_model_class(cfg['model_path'], cfg['class_name'])
    model = ModelClass().to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*STATS),
    ])
    ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    # 随机不放回抽样并逐个保存预测图，统计准确率（top-1）
    n = min(count, len(ds))
    indices = list(range(len(ds)))
    sampled = random.sample(indices, n)

    total = 0
    correct = 0
    for out_idx, ds_idx in enumerate(sampled):
        img_tensor, true_label = ds[ds_idx]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * STATS[1] + STATS[0])  # 反标准化近似显示
        img_np = (img_np.clip(0, 1) * 255).astype('uint8')
        img = Image.fromarray(img_np)
        probs, labels = classify_tensor(model, device, img_tensor.unsqueeze(0), topk=3)
        # labels 为字符串类别名，取 top-1
        pred_label_name = labels[0] if labels else ''
        try:
            pred_idx = CLASSES.index(pred_label_name)
        except ValueError:
            pred_idx = -1

        total += 1
        if pred_idx == true_label:
            correct += 1

        top_text = ", ".join([f"{labels[j]} {probs[j]*100:.1f}%" for j in range(len(labels))])
        text = f"pred: {top_text} | true: {CLASSES[true_label]}"
        annotated = annotate_image(img.resize((128, 128)), text)
        out_path = os.path.join(REPORT_IMG_DIR, f"{model_key}_test_{out_idx}.png")
        annotated.save(out_path)
        if out_idx < 5:
            print(f"[Sample] {out_idx}: {text} -> {out_path}")

    # 为避免小样本导致的误导性百分比，计算并输出对整个测试集的总体准确率（Top-1），保留更多小数位
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False,
                                         num_workers=0, pin_memory=(device.type == 'cuda'))
    all_correct = 0
    all_total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_correct += int((preds == labels).sum().item())
            all_total += labels.size(0)

    if all_total > 0:
        acc_all = all_correct / all_total * 100.0
        print(f"[Summary] Full test set Accuracy: {acc_all:.4f}% ")


def get_test_loader(batch_size: int = 512, num_workers: int = 0, pin_memory: bool = False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*STATS),
    ])
    ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    return ds, loader


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_perclass_metrics(cm: np.ndarray):
    num_classes = cm.shape[0]
    total = cm.sum()
    metrics = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / total if total > 0 else 0.0
        metrics.append({'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc})
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_perclass_metrics(metrics: List[dict], class_names: List[str], out_path: str):
    ind = np.arange(len(class_names))
    width = 0.25
    precision = [m['precision'] for m in metrics]
    recall = [m['recall'] for m in metrics]
    f1 = [m['f1'] for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ind - width, precision, width, label='Precision')
    ax.bar(ind, recall, width, label='Recall')
    ax.bar(ind + width, f1, width, label='F1')
    ax.set_xticks(ind)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_full_test(model_key: str, weights: str, device: torch.device, batch_size: int = 512, num_workers: int = 0):
    cfg = MODEL_CONFIGS[model_key]
    ModelClass = load_model_class(cfg['model_path'], cfg['class_name'])
    model = ModelClass().to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds, loader = get_test_loader(batch_size=batch_size, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(CLASSES))
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    perclass = compute_perclass_metrics(cm)

    confmat_path = os.path.join(REPORT_IMG_DIR, f"{model_key}_confmat.png")
    percls_path = os.path.join(REPORT_IMG_DIR, f"{model_key}_perclass.png")
    plot_confusion_matrix(cm, list(CLASSES), confmat_path)
    plot_perclass_metrics(perclass, list(CLASSES), percls_path)
    print(f"[Report] Acc={acc*100:.2f}% | ConfMat -> {confmat_path} | PerClass -> {percls_path}")

    # 保存Markdown报告
    md_path = os.path.join(REPORT_DIR, f"demo_report_{model_key}.md")
    rel_conf = os.path.relpath(confmat_path, REPORT_DIR).replace('\\', '/')
    rel_perc = os.path.relpath(percls_path, REPORT_DIR).replace('\\', '/')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Demo Report: {model_key}\n\n")
        f.write(f"- Weights: `{weights}`\n")
        f.write(f"- Test Accuracy: **{acc*100:.2f}%**\n\n")
        f.write(f"## Confusion Matrix\n\n")
        f.write(f"![Confusion Matrix]({rel_conf})\n\n")
        f.write(f"## Per-Class Metrics\n\n")
        f.write(f"![Per-Class Metrics]({rel_perc})\n")
    print(f"[Saved] {md_path}")


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Demo (Test-Set Only): 加载训练好的模型，在测试集样本上进行预测并保存可视化')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), required=True, help='model key')
    parser.add_argument('--weights', type=str, default='', help='path to weights (.pth), auto-search if empty')
    parser.add_argument('--dataset-sample', type=int, default=10, help='在 CIFAR-10 测试集上运行的样本数 (默认10)；设置为 10000 处理全测试集示例保存')
    parser.add_argument('--topk', type=int, default=3, help='显示Top-K预测')
    parser.add_argument('--device', type=str, default='', help='force device: cuda or cpu')
    parser.add_argument('--full', action='store_true', help='对完整测试集进行评估并生成图形化报表')
    parser.add_argument('--batch-size', type=int, default=512, help='完整评估时的批量大小')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader工作线程数（Windows建议0）')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    cfg = MODEL_CONFIGS[args.model]
    weights = args.weights or find_existing(cfg['weight_patterns'])
    if not weights:
        raise FileNotFoundError(f"Weights not found for {args.model}. Please provide --weights explicitly.")
    print(f"[Load] {args.model} <- {weights}")

    # 仅处理测试集示例保存
    if args.dataset_sample:
        run_on_test_samples(args.model, weights, device, args.dataset_sample)

    # 完整测试集评估与报表
    if args.full:
        evaluate_full_test(args.model, weights, device, batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
