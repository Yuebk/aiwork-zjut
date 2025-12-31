import os
import importlib.util
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
REPORT_IMG_DIR = os.path.join(ROOT_DIR, 'reports', 'img')
REPORT_DIR = os.path.join(ROOT_DIR, 'reports')
DATA_ROOT = os.path.join(ROOT_DIR, 'data')

os.makedirs(REPORT_IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

STATS = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

MODEL_CONFIGS = {
    'lenet': {
        'model_path': os.path.join(ROOT_DIR, 'cifar10_lenet', 'model.py'),
        'class_name': 'LeNet',
        'weight_patterns': [
            os.path.join(ROOT_DIR, 'outputs_all', 'lenet.pth'),
            os.path.join(ROOT_DIR, 'cifar10_lenet', 'outputs', 'lenet_cifar10.pth'),
            os.path.join(ROOT_DIR, 'output', 'lenet_cifar10.pth'),
            os.path.join(ROOT_DIR, 'outputs', 'lenet_cifar10.pth'),
        ],
    },
    'lenet_advanced': {
        'model_path': os.path.join(ROOT_DIR, 'cifar10_lenet_advanced', 'model.py'),
        'class_name': 'LeNetAdvanced',
        'weight_patterns': [
            os.path.join(ROOT_DIR, 'outputs_all', 'lenet_advanced.pth'),
            os.path.join(ROOT_DIR, 'cifar10_lenet_advanced', 'outputs', 'lenet_advanced_cifar10.pth'),
            os.path.join(ROOT_DIR, 'output', 'lenet_advanced_cifar10.pth'),
            os.path.join(ROOT_DIR, 'outputs', 'lenet_advanced_cifar10.pth'),
        ],
    },
    'simple_vgg': {
        'model_path': os.path.join(ROOT_DIR, 'cifar10_vgg_simple', 'model.py'),
        'class_name': 'SimpleVGG',
        'weight_patterns': [
            os.path.join(ROOT_DIR, 'outputs_all', 'simple_vgg.pth'),
            os.path.join(ROOT_DIR, 'cifar10_vgg_simple', 'outputs', 'simple_vgg_cifar10.pth'),
            os.path.join(ROOT_DIR, 'output', 'simple_vgg_cifar10.pth'),
            os.path.join(ROOT_DIR, 'outputs', 'simple_vgg_cifar10.pth'),
        ],
    },
}


def load_model_class(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("dyn_model_eval", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return getattr(module, class_name)


def get_test_loader(batch_size: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*STATS),
    ])
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    return loader


def find_existing(path_list: List[str]) -> str:
    for p in path_list:
        if os.path.exists(p):
            return p
    return ''


def evaluate_model(model: nn.Module, device: torch.device, loader) -> Tuple[float, float, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0
    num_classes = len(CLASSES)
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for t, p in zip(labels.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy()):
                confmat[t, p] += 1

    test_loss = running_loss / len(loader)
    test_acc = 100.0 * correct / total
    return test_loss, test_acc, confmat


def save_confusion_matrix(confmat: np.ndarray, model_key: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confmat, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confmat.shape[1]), yticks=np.arange(confmat.shape[0]))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {model_key}')
    plt.tight_layout()
    out_path = os.path.join(REPORT_IMG_DIR, f'confmat_{model_key}.png')
    plt.savefig(out_path)
    plt.close()
    print(f'[Saved] {out_path}')


def save_per_class_metrics(confmat: np.ndarray, model_key: str):
    # precision, recall, f1 per class
    tp = np.diag(confmat).astype(float)
    pred_sum = confmat.sum(axis=0).astype(float)
    true_sum = confmat.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum>0)
    recall = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=(precision+recall)>0)
    acc_per_class = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum>0)

    csv_path = os.path.join(REPORT_DIR, f'{model_key}_perclass_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f.write('class,precision,recall,f1,accuracy,support\n')
        for i, name in enumerate(CLASSES):
            f.write(f'{name},{precision[i]:.4f},{recall[i]:.4f},{f1[i]:.4f},{acc_per_class[i]:.4f},{int(true_sum[i])}\n')
    print(f'[Saved] {csv_path}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    loader = get_test_loader()

    summary_csv = os.path.join(REPORT_DIR, 'metrics_summary.csv')
    if not os.path.exists(summary_csv):
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            f.write('model,test_loss,test_acc\n')

    for key, cfg in MODEL_CONFIGS.items():
        ModelClass = load_model_class(cfg['model_path'], cfg['class_name'])
        weights = find_existing(cfg['weight_patterns'])
        if not weights:
            print(f'[Warn] Weights not found for {key}. Skipped.')
            continue
        print(f'[Load] {key} <- {weights}')
        model = ModelClass().to(device)
        state = torch.load(weights, map_location=device)
        model.load_state_dict(state)

        test_loss, test_acc, confmat = evaluate_model(model, device, loader)
        save_confusion_matrix(confmat, key)
        save_per_class_metrics(confmat, key)

        with open(summary_csv, 'a', newline='', encoding='utf-8') as f:
            f.write(f'{key},{test_loss:.4f},{test_acc:.2f}\n')
        print(f'[Append] {summary_csv}: {key} -> acc={test_acc:.2f}% loss={test_loss:.4f}')

    print('Done. Confusion matrices and per-class metrics saved.')


if __name__ == '__main__':
    main()
