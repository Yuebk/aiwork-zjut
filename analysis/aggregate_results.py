import os
import glob
import csv
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
REPORT_IMG_DIR = os.path.join(ROOT_DIR, 'reports', 'img')

os.makedirs(REPORT_IMG_DIR, exist_ok=True)

MODEL_KEYS = ['lenet', 'lenet_advanced', 'simple_vgg']

# 可能的目录（按优先级）
CANDIDATE_DIRS = [
    os.path.join(ROOT_DIR, 'outputs_all'),
    os.path.join(ROOT_DIR, 'outputs'),
    os.path.join(ROOT_DIR, 'output'),
    os.path.join(ROOT_DIR, 'cifar10_lenet', 'outputs'),
    os.path.join(ROOT_DIR, 'cifar10_lenet', 'output'),
    os.path.join(ROOT_DIR, 'cifar10_lenet_advanced', 'outputs'),
    os.path.join(ROOT_DIR, 'cifar10_lenet_advanced', 'output'),
    os.path.join(ROOT_DIR, 'cifar10_vgg_simple', 'outputs'),
    os.path.join(ROOT_DIR, 'cifar10_vgg_simple', 'output'),
]


def has_required_columns(csv_path: str) -> bool:
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            required = {'epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'}
            return required.issubset(set(c for c in cols if c))
    except Exception:
        return False


def find_metrics_file_for_model(model_key: str) -> str:
    # 优先匹配包含模型名与 metrics 的文件名
    patterns = [f"*{model_key}*metrics*.csv", f"*{model_key}*.csv"]
    for d in CANDIDATE_DIRS:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            for p in glob.glob(os.path.join(d, pat)):
                if has_required_columns(p):
                    return p
        # 回退：扫描目录下所有 csv，尝试列名匹配
        for p in glob.glob(os.path.join(d, '*.csv')):
            if has_required_columns(p):
                # 允许无模型名匹配但包含所需列的文件作为回退
                return p
    return ''


def load_metrics(csv_path: str) -> Dict[str, List[float]]:
    data = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['epoch'].append(int(row['epoch']))
            data['train_loss'].append(float(row['train_loss']))
            data['train_acc'].append(float(row['train_acc']))
            data['test_loss'].append(float(row['test_loss']))
            data['test_acc'].append(float(row['test_acc']))
    return data


def plot_loss_comparison(metrics_map: Dict[str, Dict[str, List[float]]]):
    plt.figure(figsize=(10, 6))
    for name, m in metrics_map.items():
        plt.plot(m['epoch'], m['test_loss'], label=f'{name} (test loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(REPORT_IMG_DIR, 'loss_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f'[Saved] {out_path}')


def plot_results_comparison(metrics_map: Dict[str, Dict[str, List[float]]]):
    names = []
    accs = []
    for name, m in metrics_map.items():
        names.append(name)
        accs.append(m['test_acc'][-1])  # final epoch test acc
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accs, color=['#6aa84f', '#3c78d8', '#e69138'])
    plt.ylabel('Accuracy (%)')
    plt.title('Final Test Accuracy Comparison')
    # add labels
    for b, v in zip(bars, accs):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{v:.2f}%', ha='center', va='bottom')
    plt.ylim(0, max(accs + [100]))
    plt.tight_layout()
    out_path = os.path.join(REPORT_IMG_DIR, 'results_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f'[Saved] {out_path}')


def main():
    metrics_map: Dict[str, Dict[str, List[float]]] = {}
    for key in MODEL_KEYS:
        path = find_metrics_file_for_model(key)
        if not path:
            print(f'[Warn] Metrics not found for {key}; searched in: {CANDIDATE_DIRS}')
            continue
        try:
            metrics_map[key] = load_metrics(path)
            print(f'[Load] {key} <- {path}')
        except Exception as e:
            print(f'[Error] Failed to load {path}: {e}')
    if not metrics_map:
        print('[Error] No metrics loaded. Please run training to produce CSV files.')
        return
    # plots
    plot_loss_comparison(metrics_map)
    plot_results_comparison(metrics_map)
    print('Done. Comparison figures saved under reports/img/.')


if __name__ == '__main__':
    main()
