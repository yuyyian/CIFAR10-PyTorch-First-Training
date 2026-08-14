# CIFAR-10 Image Classification with PyTorch

这是我的第一个完整 PyTorch 深度学习训练项目。

在学习完《深度学习入门：基于 Python 的理论与实现》和 PyTorch 基础之后，我使用 CIFAR-10 数据集独立完成了一个 CNN 图像分类任务，并通过 TensorBoard 对不同训练策略进行了对照实验。

本项目的重点不是追求很高的分类准确率，而是完整实践一次深度学习模型从数据加载、模型搭建、训练、评估到实验分析的全过程。

## 项目内容

- 使用 `torchvision.datasets.CIFAR10` 加载 CIFAR-10 数据集
- 使用 `DataLoader` 进行 mini-batch 训练
- 在 `models/cnn.py` 中统一定义 CNN 网络，并自动选择 GPU / CPU
- 使用 CrossEntropyLoss 和 SGD + Momentum 完成模型优化
- 正确切换 `model.train()` 与 `model.eval()`
- 使用 `torch.no_grad()` 完成测试集评估
- 使用 TensorBoard 记录 Loss 和 Accuracy
- 使用 `state_dict()` 保存准确率最高的模型参数
- 对 Data Augmentation 和 Batch Normalization 进行对照实验
- 根据训练曲线分析过拟合与泛化能力

## 项目结构

```text
.
├── models/
│   └── cnn.py                  # CNN 模型定义
├── checkpoints/
│   └── <experiment>/
│       └── best_model.pth      # 各实验生成的最佳模型参数
├── logs/
│   └── CIFAR10/                # TensorBoard 实验日志
├── train.py                    # 模型训练与逐轮评估
├── test.py                     # 加载最佳权重并进行测试
├── requirements.txt            # Python 依赖
└── README.md
```

`data/`、`logs/`、`checkpoints/` 等运行时产物已加入 `.gitignore`。首次运行时会自动下载 CIFAR-10，`train.py` 也会自动创建 checkpoint 目录。

## 环境与安装

本项目验证环境：

```text
Python:      3.11.15
PyTorch:     2.11.0+cu128
torchvision: 0.26.0+cu128
TensorBoard: 2.21.0
CUDA:        12.8
```

激活 Conda 环境并安装依赖：

```bash
conda activate DL
python -m pip install -r requirements.txt
```

PyTorch 的 CUDA 构建需要与本机驱动和 CUDA 环境匹配。如果需要重新安装 GPU 版本，请根据自己的平台选择相应的 PyTorch 安装命令。

## 运行方式

通过 `--experiment` 选择实验配置：

| 参数 | Data Augmentation | Batch Normalization |
| --- | :---: | :---: |
| `baseline` | 否 | 否 |
| `augmentation` | 是 | 否 |
| `bn` | 否 | 是 |
| `augmentation_bn` | 是 | 是 |

分别运行四组实验：

```bash
python train.py --experiment baseline
python train.py --experiment augmentation
python train.py --experiment bn
python train.py --experiment augmentation_bn
```

训练日志和最佳模型会按实验名分别保存。例如，`augmentation_bn` 的最佳参数位于：

```text
checkpoints/augmentation_bn/best_model.pth
```

测试对应实验的最佳模型：

```bash
python test.py --experiment augmentation_bn
```

查看 TensorBoard 曲线：

```bash
tensorboard --logdir=logs/CIFAR10
```

随后在浏览器中访问终端输出的地址，通常为 `http://localhost:6006`。

## CNN 网络结构

基础 CNN 网络：

```text
Input: 3 × 32 × 32
        │
        ▼
Conv2d(3 → 32, kernel=5, padding=2)
ReLU
MaxPool2d(2)
        │
        ▼
32 × 16 × 16
        │
        ▼
Conv2d(32 → 32, kernel=5, padding=2)
ReLU
MaxPool2d(2)
        │
        ▼
32 × 8 × 8
        │
        ▼
Conv2d(32 → 64, kernel=5, padding=2)
ReLU
MaxPool2d(2)
        │
        ▼
64 × 4 × 4
        │
        ▼
Flatten
        │
        ▼
Linear(1024 → 64)
ReLU
        │
        ▼
Linear(64 → 10)
```

BatchNorm 实验在每个卷积层后加入 Batch Normalization：

```text
Conv → BatchNorm → ReLU → Pooling
```

## 训练配置

```text
Dataset:       CIFAR-10
Batch Size:    64
Epochs:        20
Optimizer:     SGD
Learning Rate: 0.01
Momentum:      0.9
Loss Function: CrossEntropyLoss
```

训练设备自动选择：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 数据增强

Baseline 仅使用：

```python
transforms.ToTensor()
```

数据增强版本使用：

```python
transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
```

测试集不进行随机数据增强，仅使用 `ToTensor()`。

## 对照实验

为了观察 Data Augmentation 和 Batch Normalization 的影响，共进行了四组实验：

1. Baseline
2. Baseline + Data Augmentation
3. Baseline + Batch Normalization
4. Baseline + Data Augmentation + Batch Normalization

除上述变量外，其余训练条件保持一致。

### 第 20 轮训练结果

| Experiment | Train Accuracy | Test Accuracy | Train Loss | Test Loss |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 90.32% | 68.77% | 0.280 | 1.472 |
| Data Augmentation | 75.74% | 76.03% | 0.695 | 0.705 |
| BatchNorm | 96.06% | 75.13% | 0.113 | 1.188 |
| Augmentation + BatchNorm | 81.64% | **78.99%** | 0.526 | **0.640** |

> 以上结果来自单次训练。模型初始化、DataLoader shuffle 和随机数据增强都会带来随机波动，因此不能将单次准确率差异视为严格的统计结论。

## 实验分析

### Baseline

Baseline 最终训练准确率为 90.32%，测试准确率为 68.77%；训练损失下降到 0.280，而测试损失上升到 1.472。

随着训练继续进行，训练损失持续下降，测试损失却先下降后上升；训练准确率不断提高，测试准确率没有同步改善。这说明模型出现了明显的 **Overfitting（过拟合）**。

这次实验让我第一次从真实训练曲线中理解到：

> 训练集上的 Loss 越低，并不代表模型的泛化能力越好。

### Data Augmentation

加入 RandomCrop 和 RandomHorizontalFlip 后，训练准确率下降到 75.74%，但测试准确率提高到 76.03%，Train Loss 和 Test Loss 也非常接近。

训练过程中，模型不断看到经过随机裁剪和翻转后的图像，训练任务本身变得更难。与此同时，数据变化减少了模型对原始训练样本的过度拟合，提高了泛化能力。

### Batch Normalization

只加入 BatchNorm 后，训练准确率达到 96.06%，测试准确率达到 75.13%。这说明 BatchNorm 使网络更容易优化，提高了模型的训练速度和拟合能力。

不过，Train Accuracy 与 Test Accuracy 之间仍然存在较大差距。因此在本次实验中，BatchNorm 没有直接消除过拟合，而是明显增强了模型学习训练数据的能力。

### Data Augmentation + BatchNorm

两种方法结合取得了四组实验中最好的结果：

```text
Train Accuracy: 81.64%
Test Accuracy:  78.99%
Train Loss:     0.526
Test Loss:      0.640
```

可以将二者的作用简单理解为：BatchNorm 帮助模型更容易优化和学习，Data Augmentation 增加训练数据的变化、减少模型对训练样本的死记硬背。两者结合后，同时获得了较好的训练能力和泛化能力。

## TensorBoard

训练过程中记录四项指标：

```text
train_loss
test_loss
train_accuracy
test_accuracy
```

不同实验分别存放在独立的日志目录中：

```text
logs/CIFAR10/
├── baseline/
├── augmentation/
├── bn/
└── augmentation_bn/
```

这使四组实验的训练曲线可以在同一个 TensorBoard 页面中直接比较。

## Best Checkpoint

训练过程中根据测试准确率保存当前表现最好的模型：

```python
if test_accuracy > best_accuracy:
    best_accuracy = test_accuracy
    torch.save(
        net.state_dict(),
        f"checkpoints/{experiment}/best_model.pth",
    )
```

没有直接保存最后一个 Epoch 的模型，因为训练准确率可能继续提高，但测试准确率可能已经开始下降。因此，最后一轮不一定是泛化能力最好的模型，这也是 Best Checkpoint 和 Early Stopping 的意义。

当前项目直接使用测试集选择最佳 checkpoint，适合入门阶段理解完整训练流程。更规范的实验应从训练集中划分独立验证集，使用验证集选择模型，并仅在实验结束后使用测试集进行一次最终评估。

## 项目收获

这个项目最大的收获并不是将 CIFAR-10 准确率训练到接近 80%，而是第一次完整理解并实践了一个深度学习训练流程：

```text
Dataset
   ↓
DataLoader
   ↓
Model
   ↓
Forward
   ↓
Loss
   ↓
Backward
   ↓
Optimizer
   ↓
Evaluation
   ↓
TensorBoard
   ↓
Checkpoint
   ↓
Experiment Analysis
```

同时通过实际实验观察并理解了：

- Overfitting
- Generalization
- Data Augmentation
- Batch Normalization
- Train / Test Gap
- Best Checkpoint
- 控制变量实验
- Ablation Study 的基本思想

相比单纯学习 PyTorch API，这次训练让我第一次开始从“让代码运行起来”转向“观察和分析模型为什么会产生这样的结果”。

## 下一步计划

- 阅读 ResNet 论文
- 理解 Residual Connection
- 自己实现 ResNet18
- 在 CIFAR-10 上训练 ResNet
- 与当前 CNN Baseline 进行对比
