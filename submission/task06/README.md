# 任务六：验收固定尺寸的 ONNX 模型

## 导出配置

用 `yolo11n.pt` 导出固定 416×416、batch 1、无动态轴的 ONNX：

| 参数 | 值 |
| --- | --- |
| 权重 | `yolo11n.pt` |
| 格式 | `onnx` |
| `imgsz` | 416 |
| `dynamic` | `False` |
| `simplify` | `False` |
| `opset` | 12 |
| `device` | `cpu` |
| `batch` | 1 |
| `half` / `int8` | `False` / `False` |
| 输出 | `submission/task06/yolo11n-416.onnx` |
| ONNX 输入形状 | `[1, 3, 416, 416]` |

完整记录在 `onnx-export-config.json`，导出终端输出在 `export.log`。`yolo11n-416.onnx` 被 `.gitignore` 的 `*.onnx` 忽略，不随分支提交；`export_onnx.py` 可重新生成。

## 实际输入尺寸

`task06-input-sizes.csv`：

| 后端 | 实际输入形状 | 证据 |
| --- | --- | --- |
| PyTorch | `[1, 3, 416, 416]` | 对 3/3 帧注册 forward pre-hook 记录 |
| ONNX Runtime | `[1, 3, 416, 416]` | `InferenceSession.get_inputs()[0].shape` |

两边都使用相同的 `lab.predict_frame`、`rect=False`、`imgsz=416`、`conf=0.10`，只保留 `sports ball`，输出统一换算回原图坐标。`compare_onnx.py` 在运行前会校验 PyTorch 3 帧形状和 ONNX 固定形状，不对齐就报错。

## 三张图逐图对照

`task06-frames.csv` 是逐图明细，`task06-summary.csv` 是配对汇总。

| 帧 | 后端 | 结果 | 类别 | 框 `xyxy` | 置信度 | 配对 IoU | 置信度差 |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 100 | PyTorch | 有检测 | `sports ball` | `[140.13, 391.16, 260.04, 516.37]` | `0.638958` | `1.000000` | `0.000000060` |
| 100 | ONNX Runtime | 有检测 | `sports ball` | `[140.13, 391.16, 260.04, 516.37]` | `0.638958` | `1.000000` | `0.000000060` |
| 150 | PyTorch | 空 | — | — | — | — | — |
| 150 | ONNX Runtime | 空 | — | — | — | — | — |
| 200 | PyTorch | 空 | — | — | — | — | — |
| 200 | ONNX Runtime | 空 | — | — | — | — | — |

关键数值：

- 第 100 帧：两边各 1 个 `sports ball`，同类一一配对；IoU `0.9999995`（表中四舍五入为 `1.000000`），置信度差 `5.96e-8`。
- 第 150 帧：球在画面左边界，416/0.10 下两边都为空，空结果一致。
- 第 200 帧：无球，两边都为空，空结果一致。
- 没有出现新增框、消失框或偏移框；第 100 帧的框坐标差异约 `6e-5` 像素，来自后端数值误差，远小于判定门槛。

对照图：

- `rt-100-backends.png`
- `rt-150-backends.png`
- `rt-200-backends.png`

## 是否切换后端

**满足切换条件，可以切换到 ONNX Runtime 后端。**

按题面标准逐项核对：

| 条件 | 结果 |
| --- | --- |
| 两边输入尺寸一致 | PyTorch `[1,3,416,416]`，ONNX `[1,3,416,416]`，3/3 帧一致 |
| 同类框一一配对 | 第 100 帧各 1 个 `sports ball`，一一配对 |
| 配对框 IoU ≥ 0.95 | `0.9999995`，通过 |
| 置信度差 ≤ 0.01 | `5.96e-8`，通过 |
| 空结果一致 | 第 150、200 帧两边都为空，通过 |
| 增加/消失/偏移框 | 无 |

因此这份 416×416 的 ONNX 模型在 rt-100、rt-150、rt-200 上与 PyTorch 的框、类别、置信度和空结果都一致，达到切换标准。这个结论只覆盖这三张样例；部署前仍建议在更大的独立集上复核。

## 交付文件

| 文件 | 用途 |
| --- | --- |
| `onnx-export-config.json` | 导出配置与实际 ONNX 输入形状 |
| `export.log` | ONNX 导出终端记录 |
| `task06-input-sizes.csv` | 两种后端实际输入尺寸 |
| `task06-frames.csv` | 三张图逐图对照，含框、类别、置信度、空结果 |
| `task06-summary.csv` | 每帧配对与门槛结果 |
| `task06-decision.json` | 切换判断汇总 |
| `backend-comparison.json` | 仓库入口的原始逐图 JSON |
| `compare.log` | 双后端比较终端记录 |
| `rt-100-backends.png`、`rt-150-backends.png`、`rt-200-backends.png` | 双后端并排图 |
| `export_onnx.py`、`build_task06.py`、`run_task06.sh` | 复跑入口 |

## 复跑

```bash
bash submission/task06/run_task06.sh
```

等价命令：

```bash
.venv/bin/python submission/task06/export_onnx.py
.venv/bin/python compare_onnx.py --onnx submission/task06/yolo11n-416.onnx --output submission/task06/backend-comparison.json
.venv/bin/python submission/task06/build_task06.py
```

## 问卷附件对应

| 题目 | 上传文件 |
| --- | --- |
| 任务六｜导出配置与实际输入尺寸 | `onnx-export-config.json`、`task06-input-sizes.csv`；可附 `export.log` |
| 任务六｜三张图的双后端输出 | `task06-frames.csv`；可附三张 `*-backends.png` 和 `backend-comparison.json` |
| 任务六｜是否切换后端及依据 | 粘贴下面这段 |

> 满足切换条件，可以切换到 ONNX Runtime。两边实际输入都是 `[1,3,416,416]`，预处理同为 `rect=False` 的缩放补边、`conf=0.10`、只保留 `sports ball`。第 100 帧两边各 1 个同类框，框 `[140.13,391.16,260.04,516.37]`，IoU `0.9999995`，置信度都是 `0.638958`，差 `5.96e-8`；第 150、200 帧两边都为空。没有新增、消失或偏移的框，IoU 和置信度差都远低于 `0.95` 和 `0.01` 的门槛，空结果也一致。
