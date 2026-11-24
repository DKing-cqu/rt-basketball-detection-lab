# 任务二：核对两种画面尺寸的 YOLO 标注

四张图的标注统一使用类别编号 `0`、类别名 `basketball`，导出格式为 CVAT 的 `Ultralytics YOLO Detection`。

## 交付文件

| 文件 | 用途 |
| --- | --- |
| `cvat-ultralytics-yolo-detection.zip` | Ultralytics YOLO Detection 包，含 4 张原图、4 个标签文件和 `data.yaml` |
| `rt-150-restored.png` | 用导出坐标在 `rt-150.jpg` 原图（360×640）上还原的框 |
| `plos02-055-restored.png` | 用导出坐标在 `plos02-055.jpg` 原图（1920×1080）上还原的框 |
| `four-image-check.csv` | 四张图的文件名、宽高、有无标签、框坐标和归一化标签 |
| `build_task02.py` | 生成 ZIP、两张还原图和核对表的复跑脚本 |

## 在 CVAT 里完成导出

这个 ZIP 已经按 CVAT 的 `Ultralytics YOLO Detection` 结构生成，可以直接导入 CVAT 复核。若题目要求必须是 CVAT 页面导出的包，按下面步骤再导出一次：

1. 打开 CVAT 的 `Projects` 页面，新建项目，例如 `task02-basketball`。
2. 在项目右上角 `Actions` → `Import dataset`。
3. 格式选择 `Ultralytics YOLO Detection`，上传 `cvat-ultralytics-yolo-detection.zip`。
4. 导入会在项目里新建一个 `train` 任务，四张图和标签一起进去。项目没有标签时会自动从 `data.yaml` 创建 `basketball`。
5. 打开任务逐张检查：
   - `rt-125.jpg`：完整球，框贴合球边缘。
   - `rt-150.jpg`：球在左边界出画，框只框可见部分，左边界为 `x=0`。
   - `rt-200.jpg`：无球，标签为空。
   - `plos02-055.jpg`：完整球，使用 1920×1080 原图坐标。
6. 任务 `Actions` → `Export task dataset`，格式选 `Ultralytics YOLO Detection`，勾选保存图片，下载 ZIP。
7. 把 CVAT 导出的 ZIP 作为“任务二｜CVAT 导出包”上传；如果需要，也可以用它覆盖本目录的 `cvat-ultralytics-yolo-detection.zip`。

导出的目录结构应为：

```text
archive.zip/
├── data.yaml
├── train.txt
├── images/train/
│   ├── rt-125.jpg
│   ├── rt-150.jpg
│   ├── rt-200.jpg
│   └── plos02-055.jpg
└── labels/train/
    ├── rt-125.txt
    ├── rt-150.txt
    ├── rt-200.txt
    └── plos02-055.txt
```

`data.yaml`：

```yaml
path: ./
train: train.txt
names:
  0: basketball
```

标签文件是归一化的 `class_id cx cy w h`。`rt-200.txt` 是空文件，但图片仍保留在 `train.txt` 和 `images/train/` 中。

## 四张图标注核对

| 文件名 | 宽 | 高 | 有标签 | 标签数 | 框 `xyxy`（原图坐标） | 归一化标签 |
| --- | ---: | ---: | --- | ---: | --- | --- |
| `rt-125.jpg` | 360 | 640 | 有 | 1 | `48,303,167,425` | `0 0.298611 0.568750 0.330556 0.190625` |
| `rt-150.jpg` | 360 | 640 | 有 | 1 | `0,247,31,344` | `0 0.043056 0.461719 0.086111 0.151562` |
| `rt-200.jpg` | 360 | 640 | 无 | 0 | — | 空文件 |
| `plos02-055.jpg` | 1920 | 1080 | 有 | 1 | `1072,563,1141,632` | `0 0.576302 0.553241 0.035937 0.063889` |

`train` 只是这批 CVAT 数据的子集名，不代表模型训练/验证划分。

## 坐标还原方法

`build_task02.py` 不是直接拿 `annotations.csv` 画框，而是：

1. 从 ZIP 的 `labels/train/*.txt` 读出归一化坐标；
2. 用各自原图的宽高反算：
   - `x1 = (cx - w/2) * W`
   - `y1 = (cy - h/2) * H`
   - `x2 = (cx + w/2) * W`
   - `y2 = (cy + h/2) * H`
3. 在原图尺寸上画框并保存 PNG。

结果：

| 图片 | 原图尺寸 | 还原框 `xyxy` | 与人工标注最大偏差 |
| --- | --- | --- | ---: |
| `rt-150.jpg` | 360×640 | `0,247,31,344` | `0 px` |
| `plos02-055.jpg` | 1920×1080 | `1072,563,1141,632` | `0 px` |

两张 PNG 都保持原图宽高，没有缩放或补边。

## 复跑

```bash
.venv/bin/python submission/task02/build_task02.py
```

脚本会校验原图实际宽高与 `annotations.csv` 一致、ZIP 内文件名与标签一一对应、还原框与人工标注偏差不超过 1 像素。

## 问卷附件对应

| 题目 | 上传文件 |
| --- | --- |
| 任务二｜CVAT 导出包 | CVAT 导出的 ZIP；可直接用 `cvat-ultralytics-yolo-detection.zip` 导入后重新导出 |
| 任务二｜两张坐标还原图 | `rt-150-restored.png`、`plos02-055-restored.png` |
| 任务二｜四张图标注核对 | `four-image-check.csv`，或按上面的表格逐行填写 |
