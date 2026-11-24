# 任务四：比较冻结部分层与全量微调

从同一个预训练 `yolo11n.pt` 出发，用仓库的 33 张训练图和 14 张验证图训练两组，每组 5 轮。训练设置沿用仓库第三章的 `lab.py train` 入口，只改冻结范围和运行名。

## 两组配置

| 方案 | 权重 | 训练图 | 验证图 | 轮数 | 输入 | batch | 优化器 | lr0 | freeze |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 全量微调 | `yolo11n.pt` | 33 | 14 | 5 | 416 | 8 | AdamW | 0.001 | 0 |
| 冻结前 10 层 | `yolo11n.pt` | 33 | 14 | 5 | 416 | 8 | AdamW | 0.001 | 10 |

其余共同设置：`device=cpu`、`workers=0`、`seed=42`、`deterministic=True`、`mosaic=0.0`、`mixup=0.0`、`scale=0.20`、`plots=False`。完整参数见 `full-finetune-args.yaml` 和 `frozen10-args.yaml`。

## 交付文件

| 文件 | 用途 |
| --- | --- |
| `full-finetune-config.json`、`frozen10-config.json` | 两组训练配置 |
| `full-finetune-args.yaml`、`frozen10-args.yaml` | Ultralytics 实际使用的完整参数 |
| `full-finetune-results.csv`、`frozen10-results.csv` | 各自每轮记录 |
| `task04-epochs.csv` | 两组每轮记录合并表 |
| `task04-parameters.csv` | 冻结范围、可训练/冻结参数数量 |
| `full-finetune-freeze-check.json`、`frozen10-freeze-check.json` | 训练后冻结参数是否变化的核验 |
| `task04-eval.csv` | 在 640 输入、0.25 阈值、IoU 0.50 下的两版权重对照表 |
| `eval-full.jsonl`、`eval-frozen10.jsonl` | 最终评测逐帧结果 |
| `train-full.log`、`train-frozen10.log` | 两次训练的终端记录 |
| `train_task04.py`、`build_task04.py`、`run_task04.sh` | 训练、汇总和复跑入口 |

## 每轮记录

完整数据在 `task04-epochs.csv`，这里列出关键列：

| 方案 | 轮 | train box | train cls | train dfl | val box | val cls | val dfl | val mAP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全量 | 1 | 2.145 | 5.021 | 0.943 | 0.689 | 3.687 | 0.942 | 0 |
| 全量 | 2 | 2.345 | 3.092 | 1.093 | 1.032 | 4.172 | 1.269 | 0 |
| 全量 | 3 | 2.436 | 2.325 | 1.039 | 1.125 | 4.313 | 1.414 | 0 |
| 全量 | 4 | 2.129 | 1.974 | 0.944 | 1.315 | 4.500 | 1.591 | 0 |
| 全量 | 5 | 1.775 | 1.585 | 0.788 | 1.482 | 4.628 | 1.758 | 0 |
| 冻结 10 层 | 1 | 2.158 | 5.072 | 0.973 | 0.658 | 3.657 | 0.954 | 0 |
| 冻结 10 层 | 2 | 2.356 | 3.087 | 1.104 | 1.402 | 4.024 | 1.208 | 0 |
| 冻结 10 层 | 3 | 2.276 | 2.505 | 1.033 | 1.608 | 4.187 | 1.319 | 0 |
| 冻结 10 层 | 4 | 2.011 | 1.990 | 0.920 | 1.630 | 4.402 | 1.375 | 0 |
| 冻结 10 层 | 5 | 1.872 | 1.579 | 0.800 | 1.403 | 4.526 | 1.319 | 0 |

训练期验证是 Ultralytics 在训练输入 416 下自动跑的；最终评测另按题目要求用 640 输入、`conf=0.25`、`IoU>=0.50` 跑，两者不能混为一张表。

## 参数更新范围

`task04-parameters.csv` 的记录：

| 方案 | 冻结层 | 总参数 | 可训练参数 | 冻结参数 | 冻结张量数 | 冻结参数最大变化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 全量 | 无 | 2,624,080 | 2,624,064 | 16 | 1 | 0.0 |
| 冻结前 10 层 | `model.0`–`model.9` | 2,624,080 | 1,508,320 | 1,115,760 | 100 | 0.0 |

说明：

- Ultralytics 在两个方案里都会固定冻结 Detect 头的 `.dfl` 参数（16 个），这是框架默认行为；所以“全量”是指除 DFL 外的参数都参与更新。
- 冻结前 10 层方案里，`model.0`–`model.9` 的 1,115,744 个参数加 DFL 共 1,115,760 个参数不接收梯度，剩下 1,508,320 个参数更新。
- 两组训练后冻结参数的最大绝对变化都是 `0.0`，说明冻结确实生效。
- 冻结是设置 `requires_grad=False`，这些可学习参数不接收反向传播梯度；BatchNorm 的 `running_mean`/`running_var` 是 buffer，不是可学习参数，训练模式下仍可能更新。
- 训练集 33 张参与前向、反向传播和参数更新；验证集 14 张只用于每轮评估和 `best.pt` 选择，不参与梯度更新。

## 固定口径评测

`task04-eval.csv`，两版权重都在 `imgsz=640`、`conf=0.25`、`IoU>=0.50`、`split=val` 下评测：

| 方案 | 完整球匹配 | 边缘球匹配 | 无球误报 | 全部有球匹配 |
| --- | ---: | ---: | ---: | ---: |
| 全量微调 | `0/10` | `0/2` | `0/2` | `0/12` |
| 冻结前 10 层 | `0/10` | `0/2` | `0/2` | `0/12` |

两版权重在 640/0.25 下都没有产生可通过阈值和 IoU 门槛的检测。`best.pt` 的挑选基于训练期 mAP50，而两组每轮 mAP50 都是 0，因此 `best.pt` 本身也没有区分度。

## 判断

**现有证据不足，暂时无法判断冻结前 10 层是否更值得继续。**

原因：

1. 两组在固定评测口径下完全相同：完整球 `0/10`、边缘球 `0/2`、无球误报 `0/2`，没有一方更好。
2. 两组每轮训练期 `val mAP50` 都是 0；训练损失虽然下降，但还没有形成可用的检测输出。
3. 5 轮里 `warmup_epochs=3.0` 占了 3 轮，实际有效学习轮数很少；33 张图每轮只有 5 个 batch，5 轮约 25 次参数更新，不足以让重新初始化的 1 类检测头收敛。
4. Ultralytics 的 `best.pt` 保存的是 EMA 权重；约 25 次更新时 EMA 仍接近初始值，所以保存下来的权重置信度也很低。
5. 冻结前 10 层只更新 1,508,320 个参数，训练成本更低；但在没有可用检测的情况下，这只能说明它更省，不能说明它更准。

下一步应保持两组冻结范围不变，先把训练轮数延长（例如 30–50 轮），或调整学习率/warmup，让检测头收敛，再用同一套 640/0.25/IoU 0.50 口径比较。当前 5 轮结果不足以支持“冻结更好”或“全量更好”的结论。

## 复跑

```bash
bash submission/task04/run_task04.sh
```

等价命令：

```bash
.venv/bin/python submission/task04/train_task04.py --scheme full
.venv/bin/python submission/task04/train_task04.py --scheme frozen10
.venv/bin/python lab.py eval --weights runs/task04-full/weights/best.pt --split val --imgsz 640 --conf 0.25 --output submission/task04/eval-full.jsonl
.venv/bin/python lab.py eval --weights runs/task04-frozen10/weights/best.pt --split val --imgsz 640 --conf 0.25 --output submission/task04/eval-frozen10.jsonl
.venv/bin/python submission/task04/build_task04.py
```

`runs/` 被 `.gitignore` 忽略，权重不随分支提交；配置、每轮记录、冻结核验和评测结果都在 `submission/task04/`。

## 问卷附件对应

| 题目 | 上传文件 |
| --- | --- |
| 任务四｜两组训练配置和每轮记录 | `full-finetune-config.json`、`frozen10-config.json`、`task04-epochs.csv`；可附两个 `results.csv` 和 `args.yaml` |
| 任务四｜两版权重评测对照表 | `task04-eval.csv`；可附 `task04-parameters.csv` 和两个逐帧 JSONL |
| 任务四｜参数更新与数据分工 | 粘贴下面第一段 |
| 任务四｜微调方案取舍 | 粘贴下面第二段 |

> 全量微调方案中，除 Ultralytics 固定冻结的 Detect 头 DFL（16 个参数）外，2,624,064 个参数都接收梯度。冻结前 10 层方案中，`model.0`–`model.9` 的 1,115,744 个参数被冻结，加上 DFL 共 1,115,760 个参数不更新，剩下 1,508,320 个参数更新；训练后冻结参数最大绝对变化为 0.0。训练集 33 张参与前向、反向传播和参数更新；验证集 14 张只用于每轮评估和 `best.pt` 选择，不参与梯度更新。

> 现有证据不足，暂时无法判断冻结前 10 层是否更值得继续。两组在 640 输入、0.25 阈值、IoU 0.50 下都是完整球 0/10、边缘球 0/2、无球误报 0/2，训练期 5 轮的 val mAP50 也全是 0。两组训练损失都下降，冻结前 10 层只更新 1,508,320 个参数、成本更低，但没有可用检测，不能据此说冻结更好。5 轮里 warmup 占 3 轮，33 张图约 25 次参数更新，不足以让 1 类检测头收敛。下一步应延长训练轮数或调整学习率/warmup，再用同一口径比较；当前结果不足以支持取舍。
