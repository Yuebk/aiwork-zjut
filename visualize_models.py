import torch
import torch.nn as nn
import os
import sys
import importlib.util
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchviz import make_dot
import graphviz

# 基础路径配置
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_CONFIGS = {
    'lenet': {
        'model_path': os.path.join(BASE_DIR, 'cifar10_lenet', 'model.py'),
        'class_name': 'LeNet',
    },
    'lenet_advanced': {
        'model_path': os.path.join(BASE_DIR, 'cifar10_lenet_advanced', 'model.py'),
        'class_name': 'LeNetAdvanced',
    },
    'simple_vgg': {
        'model_path': os.path.join(BASE_DIR, 'cifar10_vgg_simple', 'model.py'),
        'class_name': 'SimpleVGG',
    },
}

def load_model_class(model_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("dyn_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def export_visualizations():
    output_dir = os.path.join(BASE_DIR, 'model_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # CIFAR-10 的虚拟输入: Batch size 1, 3 channels, 32x32
    dummy_input = torch.randn(1, 3, 32, 32)
    
    print(f"Exporting visualizations to: {output_dir}")
    
    for key, config in MODEL_CONFIGS.items():
        print(f"\nProcessing {key}...")
        
        try:
            # 加载模型
            ModelClass = load_model_class(config['model_path'], config['class_name'])
            model = ModelClass()
            model.eval()
            
            # 1. 导出 ONNX (用于 Netron 查看)
            onnx_path = os.path.join(output_dir, f"{key}.onnx")
            torch.onnx.export(
                model, 
                dummy_input, 
                onnx_path, 
                input_names=['input'], 
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                opset_version=12
            )
            print(f"  [ONNX] Saved to: {onnx_path}")
            
            # 2. 导出 TensorBoard Graph
            log_dir = os.path.join(output_dir, 'tensorboard_logs', key)
            writer = SummaryWriter(log_dir=log_dir)
            writer.add_graph(model, dummy_input)
            writer.close()
            print(f"  [TensorBoard] Saved logs to: {log_dir}")

            # 3. Torchinfo 结构摘要
            summary_txt_path = os.path.join(output_dir, f"{key}_torchinfo.txt")
            info = summary(
                model,
                input_size=(1, 3, 32, 32),
                col_names=("input_size", "output_size", "num_params", "kernel_size"),
                verbose=0,
            )
            with open(summary_txt_path, 'w', encoding='utf-8') as f:
                f.write(str(info))
            print(f"  [Torchinfo] Saved to: {summary_txt_path}")

            # 4. Torchviz 计算图导出
            torchviz_base = os.path.join(output_dir, f"{key}_torchviz")
            try:
                with torch.no_grad():
                    output = model(dummy_input)
                dot = make_dot(output, params=dict(model.named_parameters()))
                dot.format = 'png'
                dot.render(filename=torchviz_base, cleanup=True)
                print(f"  [Torchviz] Saved PNG to: {torchviz_base}.png")
            except Exception as viz_err:
                # 回退保存为 .dot 源文件，便于手动用 Graphviz 渲染
                try:
                    with torch.no_grad():
                        output = model(dummy_input)
                    dot = make_dot(output, params=dict(model.named_parameters()))
                    dot_path = f"{torchviz_base}.dot"
                    with open(dot_path, 'w', encoding='utf-8') as f:
                        f.write(dot.source)
                    print(f"  [Torchviz] PNG render failed ({viz_err}); saved DOT to: {dot_path}")
                except Exception as dot_err:
                    print(f"  [Torchviz] Failed to generate DOT: {dot_err}")
            
        except Exception as e:
            print(f"  Error processing {key}: {e}")

    print("\nDone!")
    print("-" * 50)
    print("查看方式:")
    print("1. Netron (推荐): 打开 https://netron.app/ 并加载生成的 .onnx 文件")
    print(f"2. TensorBoard: 运行命令 tensorboard --logdir={os.path.join(output_dir, 'tensorboard_logs')}")
    print("3. Torchinfo: 查看 *_torchinfo.txt 以获取详细层级与参数统计")
    print("4. Torchviz: 直接查看 *_torchviz.png；若未生成 PNG，请用 Graphviz 渲染 *_torchviz.dot")

if __name__ == "__main__":
    export_visualizations()
