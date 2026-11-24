# 任务一：配置环境并跑通篮球检测

所有命令均从仓库根目录执行，解释器固定为项目 `.venv`：

```bash
.venv/bin/python --version
```

## 交付文件

| 文件 | 用途 |
| --- | --- |
| `env.txt` | 环境核验日志：解释器路径、依赖版本、CPU 核验 |
| `env-check.sh` | 生成 `env.txt` 的复跑脚本 |
| `rt-100-box.png` | `rt-100.jpg` 的带框图（`conf=0.25`） |
| `rt100-inference.json` | 首次推理记录：权重 SHA-256、`cpu`、`imgsz=640`、`conf=0.25`、检测框、中心点、复跑命令 |
| `rt-225.jpg` | 从 `dataset-curated/images/val/rt-225.jpg` 原样复制的原图 |
| `rt-225-flip.png` | 由 `rt-225.jpg` 水平翻转得到的 PNG |
| `rt-225-center-comparison.png` | 原图 / 翻转图的中心点合图 |
| `rt-225-centers.json` | 两张图的检测框、最高分中心、镜像回原图坐标 |
| `replay-conf010.jsonl` | RT 视频 302 帧逐帧回放（`conf=0.10`） |
| `replay-frames-200-225.jsonl` | 第 200、225 帧的回放记录 |
| `replay-meta.json` | 回放权重、设备、输入尺寸、阈值、命令 |
| `results.csv` | 问卷用汇总表 |
| `build_task01.py` | 生成上述图片、JSON、CSV 的复跑脚本 |

`runs/` 被 `.gitignore` 忽略，因此回放日志直接写在 `submission/task01/`，随分支交付。目录中的 `log_conf010.txt`、`log_conf025.txt` 是早先试跑留下的日志；验收以 `replay-conf010.jsonl` 和 `replay-frames-200-225.jsonl` 为准。

## 环境核验

```bash
bash submission/task01/env-check.sh | tee submission/task01/env.txt
```

关键记录：

```text
sys.executable: /home/dking/rt-basketball-detection-lab/rt-basketball-detection-lab/.venv/bin/python
python: 3.12.3
PyTorch: 2.6.0+cpu
torchvision: 0.21.0+cpu
Ultralytics: 8.3.232
OpenCV: 4.11.0 (opencv-python 4.11.0.86)
NumPy: 1.26.4
ONNX: 1.19.1
ONNX Runtime: 1.32.0
torch.version.cuda: None
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

依赖安装顺序（与仓库 README 一致；`onnx` 固定在第六题的 `1.19.1`）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements.txt
python -m pip install -c requirements.txt onnx==1.19.1 onnxruntime==1.32.0
```

## 复跑入口

```bash
# 1. 准备视频、权重和固定帧；已准备过会复用并校验
.venv/bin/python lab.py prepare

# 2. 回放 RT 视频，只保留 sports ball，CPU，640 输入，conf=0.10
.venv/bin/python lab.py replay \
  --source data-src/rt.mp4 \
  --imgsz 640 \
  --conf 0.10 \
  --output submission/task01/replay-conf010.jsonl

# 3. 生成带框图、翻转对照图、中心点 JSON 和汇总表
.venv/bin/python submission/task01/build_task01.py
```

`build_task01.py` 会校验回放日志中第 200 帧检测为空、第 225 帧有检测；条件不满足时直接报错，不会生成看似通过的表格。

## 结果

### 首次推理：`rt-100.jpg`，`conf=0.25`

| 项目 | 值 |
| --- | --- |
| 权重 | `yolo11n.pt`（SHA-256 `0ebbc80d...7644ee1`） |
| 设备 / 输入 / 类别 | `cpu` / 640 / `sports ball` |
| 检测框 `xyxy` | `[139.41, 390.48, 260.47, 516.74]` |
| 置信度 | `0.512906` |
| 最高分框中心 | `(199.94, 453.61)` |

框落在篮球上，见 `rt-100-box.png`。参数与复跑命令记录在 `rt100-inference.json`。

### 第 225 帧：原图与水平翻转，`conf=0.10`

| 图像 | 检测框 `xyxy` | 置信度 | 中心 `(x, y)` | 镜像回原图 `x` |
| --- | --- | --- | --- | --- |
| `rt-225.jpg` | `[24.39, 156.21, 133.00, 265.45]` | `0.251331` | `(78.69, 210.83)` | — |
| `rt-225-flip.png` | `[227.55, 156.48, 336.28, 266.08]` | `0.280921` | `(281.92, 211.28)` | `78.08` |

翻转图的中心按 `x_original = 360 - x_flipped` 回到原图坐标，得到 `(78.08, 211.28)`，与原图中心相差 `(-0.61, +0.45)` 像素。球从左半边换到右半边，中心点随框换边；坐标可由对应检测框直接复算。

### 第 200、225 帧回放，`conf=0.10`

| 帧 | 时间 `s` | 检测 | 说明 |
| --- | --- | --- | --- |
| 200 | `3.3325` | `[]` | 保留空记录，无球 |
| 225 | `3.7491` | `sports ball`，`0.193709`，`[24.64, 156.35, 132.91, 265.46]` | 有球 |

回放读取的是 `rt.mp4` 解码帧；单图对照读取的是 `lab.py prepare` 写出的 JPEG。两者像素不完全相同，所以第 225 帧的置信度分别是 `0.193709`（视频回放）和 `0.251331`（`rt-225.jpg`），检测框位置基本一致。

## 观察题作答

**同一个卷积核怎样在新位置提取局部线索？** 卷积核的权重在整张特征图上共享，它像一个固定的小窗口滑过每个位置；球移动到新位置后，同样的边缘、弧线和纹理组合在新坐标被同一组权重激活，因此响应跟着球平移，不需要为每个位置单独学习参数。左右翻转相当于把水平方向的局部线索镜像，响应位置也镜像到 `x' = W - x`。

**两张有球图的置信度是否一样？** 不一样。`rt-100.jpg` 是 `0.512906`；`rt-225.jpg` 原图是 `0.251331`，翻转图是 `0.280921`。翻转会改变像素插值和球周围的局部上下文，所以同一权重给出的分数只是接近，不会完全相同。

## 问卷附件对应

| 题目 | 上传文件 |
| --- | --- |
| 任务一｜环境核验记录 | `env.txt`（或截图终端输出）；可附 `env-check.sh` |
| 任务一｜第一次 YOLO11n 推理 | `rt-100-box.png`、`rt100-inference.json` |
| 任务一｜第 225 帧中心点对照图 | `rt-225-center-comparison.png`；原图 `rt-225.jpg`、翻转图 `rt-225-flip.png`、`rt-225-centers.json` |
| 任务一｜第 200、225 帧回放记录 | `replay-frames-200-225.jsonl`；完整日志 `replay-conf010.jsonl`、`replay-meta.json`、`results.csv` |
| 任务一｜卷积与置信度观察 | 粘贴上面「观察题作答」两段文字 |
