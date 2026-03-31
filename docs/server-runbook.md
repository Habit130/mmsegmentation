# IP102 单卡 4090 批量训练运行手册

## 目标

该仓库已整理为 Linux 单卡 RTX 4090 服务器使用形态。目标流程是：

1. 安装环境
2. 生成固定 train/val 划分
3. 顺序训练 8 个模型
4. 用验证集最佳 checkpoint 在 test.txt 上做最终测试
5. 汇总每个模型的 IoU_fg / Dice_fg / Recall_fg / mIoU / mAcc

## 目录约定

- 仓库根目录：mmsegmentation
- 数据集目录：仓库同级 ../IP102
- 数据集关键目录：
  - ../IP102/JPEGImages
  - ../IP102/Masks
  - ../IP102/ImageSets/Main/trainval.txt
  - ../IP102/ImageSets/Main/test.txt
- 运行产物：
  - work_dirs/ip102/<config_stem>/
  - results/summary.csv
  - results/summary.md

## 1. 创建环境

服务器需预装：

- NVIDIA 驱动
- 可联网安装 Python 依赖

执行：

```bash
bash server/setup.sh
source "$HOME/.local/miniforge3/etc/profile.d/conda.sh"
conda activate "$PWD/.conda-env/mmseg-1.2.2-cu121"
```

如果你在 setup.sh 中自定义了环境路径，激活时把上面的路径替换成对应值。

## 2. 一键顺序训练和测试

激活环境后执行：

```bash
bash server/train_all.sh
```

默认行为：

- 从 trainval.txt 固定切出 train.txt 和 val.txt
- 切分参数：seed=3407，val_ratio=0.1
- 使用 test.txt 做最终测试
- 顺序训练以下 8 个模型：
  - UPerNet Swin-T
  - OCRNet HRNet-W18
  - HRNet HRNet-W18
  - Segmenter ViT-B
  - FCN ResNet-50
  - PSPNet ResNet-50
  - DeepLabV3+ ResNet-50
  - SegFormer MiT-B2

## 3. 断点续跑

再次执行同一命令即可：

```bash
bash server/train_all.sh
```

脚本行为：

- 若某模型已存在 final_test_metrics.json，直接跳过
- 若训练过程中断，会对该模型使用 tools/train.py --resume
- 最后重新汇总 results/summary.csv 和 results/summary.md

## 4. 可选参数

若数据集不在默认同级路径：

```bash
DATA_ROOT=/abs/path/to/IP102 bash server/train_all.sh
```

若需要调整验证集比例或随机种子：

```bash
VAL_RATIO=0.2 SPLIT_SEED=1234 bash server/train_all.sh
```

若需要调整输出目录：

```bash
WORK_ROOT=/abs/path/work_dirs/ip102 RESULTS_DIR=/abs/path/results bash server/train_all.sh
```

## 5. 指标定义

最终汇总表固定输出：

- IoU_fg：前景类 foreground 的 IoU
- Dice_fg：前景类 foreground 的 Dice
- Recall_fg：前景类 foreground 的 Recall
- mIoU：背景和前景两类平均 IoU
- mAcc：背景和前景两类平均 Acc

所有最终指标都来自 test.txt 的最终测试，不参与训练期模型选择。

## 6. 关键脚本

- server/setup.sh：创建和安装环境
- server/train_all.sh：顺序训练、测试、汇总
- tools/ip102/generate_splits.py：生成固定 train/val
- tools/ip102/find_best_checkpoint.py：定位最佳 checkpoint
- tools/ip102/test_and_dump.py：运行最终测试并导出结构化结果
- tools/ip102/collect_results.py：汇总为 CSV/Markdown
- tools/ip102/validate_configs.py：校验 8 个配置可 build

## 7. 常见故障

- ModuleNotFoundError: mmcv/mmengine/torch
  - 说明环境未激活或安装失败，重新执行 bash server/setup.sh 后再激活环境。
- 找不到 ../IP102
  - 检查仓库与数据集是否为同级目录，或通过 DATA_ROOT=/abs/path/to/IP102 指定。
- 没有 best_mIoU checkpoint
  - 脚本会自动回退到 last_checkpoint 或 latest.pth；若都没有，说明训练未成功完成。
