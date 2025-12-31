import os
import csv
import glob
from typing import Dict, List
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
REPORT_IMG_DIR = os.path.join(ROOT_DIR, 'reports', 'img')

os.makedirs(REPORT_IMG_DIR, exist_ok=True)

MODEL_KEYS = ['lenet', 'lenet_advanced', 'simple_vgg']
CANDIDATE_DIRS = [
    os.path.join(ROOT_DIR, 'outputs_all'),
    os.path.join(ROOT_DIR, 'outputs'),
    os.path.join(ROOT_DIR, 'output'),
    os.path.join(ROOT_DIR, 'cifar10_lenet', 'outputs'),
    os.path.join(ROOT_DIR, 'cifar10_lenet_advanced', 'outputs'),
    os.path.join(ROOT_DIR, 'cifar10_vgg_simple', 'outputs'),
]


def has_required_columns(csv_path: str) -> bool:
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            required = {'epoch', 'test_acc'}
            return required.issubset(set(c for c in cols if c))
    except Exception:
        return False


def find_metrics_file_for_model(model_key: str) -> str:
    patterns = [f"*{model_key}*metrics*.csv", f"*{model_key}*.csv"]
    for d in CANDIDATE_DIRS:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            for p in glob.glob(os.path.join(d, pat)):
                if has_required_columns(p):
                    return p
        for p in glob.glob(os.path.join(d, '*.csv')):
            if has_required_columns(p):
                return p
    return ''


def load_metrics(csv_path: str):
    data = {'epoch': [], 'test_acc': []}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['epoch'].append(int(row['epoch']))
            data['test_acc'].append(float(row['test_acc']))
    return data


def main():
    metrics_map: Dict[str, Dict[str, List[float]]] = {}
    for key in MODEL_KEYS:
        path = find_metrics_file_for_model(key)
        if not path:
            print(f'[Warn] Metrics not found for {key}.')
            continue
        metrics_map[key] = load_metrics(path)
        print(f'[Load] {key} <- {path}')

    if not metrics_map:
        print('[Error] no metrics loaded.')
        return

    plt.figure(figsize=(10, 6))
    for name, m in metrics_map.items():
        plt.plot(m['epoch'], m['test_acc'], label=f'{name}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(REPORT_IMG_DIR, 'acc_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f'[Saved] {out_path}')


if __name__ == '__main__':
    main()
