# 无人机 RGB-IR 小目标检测研究工程

本工程服务于硕士论文“面向复杂场景的无人机小目标检测方法研究”，研究主线固定为：

1. 面向昼夜和低照度的可靠性自适应 RGB-IR 特征融合；
2. 浅层双流、融合后单流的轻量学生网络与跨模态知识蒸馏。

多尺度特征金字塔仅作为基础检测器组件，不作为独立创新点。

## 目录职责

- `src/rgbt_uav/`：数据协议、模型模块、蒸馏和指标台账。
- `configs/`：基线、教师、学生和消融实验配置。
- `tests/`：无需真实数据集即可执行的行为测试。
- `docs/`：研究协议、时间表、论文结构和投稿策略。
- `data/`、`runs/`、`checkpoints/`：本地生成目录，均被 Git 忽略。

## 环境

正式训练环境固定为 Python 3.10、PyTorch 2.7.1、CUDA 12.8。RTX 5060 8GB只用于小样本调试；正式实验使用24GB以上云GPU。

```powershell
conda env create -f environment.yml
conda activate rgbt-uav-thesis
pytest
rgbt-smoke-model
```

数据下载完成后，按 `docs/data-layout.md` 放置文件，再执行清单生成与审计命令。禁止把原始数据、权重和实验输出提交到 Git。

当前模型代码实现并测试了双输入接口、P3/P4/P5教师门控、P3后单流学生、特征与输出蒸馏和复杂度统计。它是正式训练适配器的可验证研究契约；真实DroneVehicle训练开始前，还需将这些模块接入锁定版本的Ultralytics检测损失、数据加载器和权重初始化，不能把随机张量冒烟结果当成论文实验。

## 最小数据闭环

```powershell
rgbt-build-manifest --format dronevehicle-xml `
  --rgb-dir data/raw/dronevehicle/train/trainimg `
  --ir-dir data/raw/dronevehicle/train/trainimgr `
  --annotation-dir data/raw/dronevehicle/train/trainlabel `
  --split train --output data/processed/manifests/train.jsonl `
  --class-alias "feright car=freight_car" `
  --class-alias "feright_car=freight_car" `
  --class-alias "feright=freight_car" `
  --class-alias "truvk=truck" --ignore-name "*" --ignore-degenerate
rgbt-audit-manifest --manifest data/processed/manifests/train.jsonl
```

首次真实运行前必须人工确认DroneVehicle发布版本的目录名、配对命名和两套模态标注坐标关系；不能仅凭脚本无报错就认定配准正确。

## 复现原则

每次实验必须记录配置文件、Git提交、数据清单哈希、随机种子、硬件、指标和GPU小时。论文表格只能由保存的实验记录生成，不能手工改写数值。
