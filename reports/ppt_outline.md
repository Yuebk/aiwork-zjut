# 10分钟答辩PPT大纲（CIFAR-10 图像分类对比）

时长：约 10 分钟（每页 45-90 秒）

---

## 1. 封面（0:45）
- 标题：CIFAR-10 图像分类：LeNet 与 VGG 风格对比
- 姓名、日期：2025-12-24
- 机构/课程信息

图片：无

---

## 2. 问题定义与目标（0:45）
- 任务：在 CIFAR-10（10 类，3×32×32）上进行分类
- 模型：
  - 基础 LeNet
  - LeNet Advanced（Conv→BN→ReLU→Pool）
  - 简化 VGG-style（3×3 堆叠 + BN + 模块化 + Pool）
- 目标：在统一配置下对比收敛、泛化与最终指标

图片：无

---

## 3. 数据与预处理（1:00）
- 数据：官方训练 50,000、测试 10,000；建议训练/验证 90/10
- 预处理：`ToTensor` + 通道标准化（均值/方差）
- 增强（训练集）：RandomCrop(32,pad=4) + RandomHorizontalFlip

图片：
- 类别分布：`reports/img/class_distribution.png`
- 通道统计：`reports/img/channel_stats.png`

---

## 4. 模型架构：LeNet（1:00）
- 结构：Conv(3→6,5×5)→ReLU→Pool→Conv(6→16,5×5)→ReLU→Pool→FC(400→120→10)
- 参数规模与输出尺寸（引用 torchinfo）

图片（任选其一或同时放置）：
- Netron：`model_visualizations/lenet.onnx`
- Torchviz：`model_visualizations/lenet_torchviz.png`
- Torchinfo摘要：`model_visualizations/lenet_torchinfo.txt`

---

## 5. 模型架构：LeNet Advanced（1:00）
- 结构改进：每个卷积后加入 BN，顺序 Conv→BN→ReLU→Pool，其余一致
- 预期效果：稳定训练、加快收敛

图片：
- Netron：`model_visualizations/lenet_advanced.onnx`
- Torchviz：`model_visualizations/lenet_advanced_torchviz.png`
- Torchinfo摘要：`model_visualizations/lenet_advanced_torchinfo.txt`

---

## 6. 模型架构：简化 VGG-style（1:00）
- Block1： [Conv3×3→BN→ReLU]×2 → Pool（输出 32×16×16）
- Block2： [Conv3×3→BN→ReLU]×2 → Pool（输出 64×8×8）
- Block3： [Conv3×3→BN→ReLU]×2 → Pool（输出 128×4×4）
- FC：128×4×4→512→Dropout→10

图片：
- Netron：`model_visualizations/simple_vgg.onnx`
- Torchviz：`model_visualizations/simple_vgg_torchviz.png`
- Torchinfo摘要：`model_visualizations/simple_vgg_torchinfo.txt`

---

## 7. 训练配置（0:45）
- 统一超参数：BS=64、LR=1e-3、Epoch=10、Adam、CrossEntropy
- 加速：AMP、`cudnn.benchmark=True`、DataLoader 并发 & pin_memory
- 并行训练：多进程并行（Windows 默认降低 num_workers，支持环境变量覆盖）
- 早停策略（可选）：patience=5、保存最优模型

图片：无（可补一张超参数表格截图）

---

## 8. 结果对比（2:00）
- 收敛曲线（训练/测试）：
  - LeNet：`cifar10_lenet/outputs/lenet_curves.png`
  - LeNet Advanced：`cifar10_lenet_advanced/outputs/lenet_advanced_curves.png`
  - Simple VGG：`cifar10_vgg_simple/outputs/simple_vgg_curves.png`
- 指标对比（Top-1 Acc & Loss）：引用训练后统计
  - 若使用并行脚本：`outputs_all/lenet_metrics.csv`、`outputs_all/lenet_advanced_metrics.csv`、`outputs_all/simple_vgg_metrics.csv`

推荐合成图（可选）：
- 结果对比：`reports/img/results_comparison.png`（柱状图/折线图聚合）
- 损失对比：`reports/img/loss_comparison.png`

---

## 9. 分析与讨论（1:00）
- BN 与数据增强对训练稳定性/泛化的影响
- 3×3 堆叠相较 5×5 的参数效率与表达能力
- 过拟合迹象与正则化（Dropout、增强）的权衡

图片：可复用曲线片段或混淆矩阵（如后续生成）

---

## 10. 结论与展望（0:45）
- 简化 VGG 在统一配置下通常表现更优（需以实测为准）
- 后续：更长训练、更强增强、学习率调度、早停/模型集成、混淆矩阵与 per-class 指标

图片：无

---

## 附录与Q&A（0:30）
- 代码路径与运行命令
- 可视化生成脚本：`visualize_models.py`
- 数据分析脚本：`analysis/data_analysis.py`

图片：可视化资源入口页（文件夹截图）

---

## 资源准备与生成说明
- 架构图与摘要：
```powershell
pip install onnx tensorboard torchinfo torchviz graphviz
python visualize_models.py
```
- 数据可视化：
```powershell
python analysis/data_analysis.py
```
- 训练曲线与指标：
```powershell
# 个别训练
python cifar10_lenet/main.py
python cifar10_lenet_advanced/main.py
python cifar10_vgg_simple/main.py

# 并行训练（重资源）
python train_all.py
```
- 若需合成对比图（可选），我可提供脚本读取 `outputs_all/*.csv` 并生成 `reports/img/results_comparison.png`。
