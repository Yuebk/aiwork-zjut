import os
import csv
import glob
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
REPORT_DIR = os.path.join(ROOT_DIR, 'reports')
IMG_DIR = os.path.join(REPORT_DIR, 'img')

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(REPORT_DIR, 'metrics_summary.csv')
SUMMARY_MD = os.path.join(REPORT_DIR, 'summary.md')
SUMMARY_HTML = os.path.join(REPORT_DIR, 'summary.html')


def load_metrics_summary(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def find_images(patterns):
    found = []
    for pat in patterns:
        found.extend(sorted(glob.glob(os.path.join(IMG_DIR, pat))))
    return found


def rel(path):
    # make path relative to REPORT_DIR for markdown
    return os.path.relpath(path, REPORT_DIR).replace('\\', '/')


def build_markdown(summary_rows, images_map):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append(f"# 实验汇总报告\n")
    lines.append(f"生成时间：{ts}\n")

    # Metrics table
    lines.append("## 测试集指标汇总\n")
    if summary_rows:
        lines.append("| 模型 | Test Acc (%) | Test Loss |\n|---|---:|---:|")
        for r in summary_rows:
            lines.append(f"| {r.get('model','')} | {float(r.get('test_acc','0')):.2f} | {float(r.get('test_loss','0')):.4f} |")
    else:
        lines.append("未找到 metrics_summary.csv，跳过表格。\n")
    lines.append("")

    # Overall comparisons
    lines.append("## 结果对比图\n")
    for title, paths in images_map.items():
        if not paths:
            continue
        lines.append(f"### {title}")
        for p in paths:
            lines.append(f"![]({rel(p)})")
        lines.append("")

    # Confusion matrices section
    lines.append("## 混淆矩阵\n")
    for p in find_images(["confmat_*.png"]):
        lines.append(f"![]({rel(p)})")
    lines.append("")

    return "\n".join(lines)


def build_html(summary_rows, images_map):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    def img_tag(p):
        return f'<img src="{rel(p)}" style="max-width:100%;height:auto;margin:8px 0;" />'

    rows_html = "".join(
        f"<tr><td>{r.get('model','')}</td><td style='text-align:right'>{float(r.get('test_acc','0')):.2f}</td><td style='text-align:right'>{float(r.get('test_loss','0')):.4f}</td></tr>"
        for r in summary_rows
    )
    table_html = (
        "<table><thead><tr><th>模型</th><th>Test Acc (%)</th><th>Test Loss</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>" if summary_rows else "<p>未找到 metrics_summary.csv，跳过表格。</p>"
    )

    blocks = []
    for title, paths in images_map.items():
        if not paths:
            continue
        imgs = "".join(img_tag(p) for p in paths)
        blocks.append(f"<h3>{title}</h3>{imgs}")

    confmats = "".join(img_tag(p) for p in find_images(["confmat_*.png"]))

    html = f"""
<!doctype html>
<html lang="zh-cn">
<head>
<meta charset="utf-8" />
<title>实验汇总报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif; margin: 24px; }}
h1, h2, h3 {{ margin: 12px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; }}
th {{ background: #f6f8fa; text-align: left; }}
.section {{ margin-top: 16px; }}
</style>
</head>
<body>
<h1>实验汇总报告</h1>
<p>生成时间：{ts}</p>
<div class="section">
  <h2>测试集指标汇总</h2>
  {table_html}
</div>
<div class="section">
  <h2>结果对比图</h2>
  {''.join(blocks)}
</div>
<div class="section">
  <h2>混淆矩阵</h2>
  {confmats}
</div>
</body>
</html>
"""
    return html


def main():
    summary_rows = load_metrics_summary(SUMMARY_CSV)
    images_map = {
        '最终测试准确率（柱状图）': find_images(['results_comparison.png']),
        '测试损失对比（折线）': find_images(['loss_comparison.png']),
        '测试准确率对比（折线）': find_images(['acc_comparison.png']),
    }

    md = build_markdown(summary_rows, images_map)
    with open(SUMMARY_MD, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'[Saved] {SUMMARY_MD}')

    html = build_html(summary_rows, images_map)
    with open(SUMMARY_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'[Saved] {SUMMARY_HTML}')


if __name__ == '__main__':
    main()
