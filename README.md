# RT 视觉组｜篮球单帧检测练习

本仓库对应飞书[《神经网络基础》](https://gcnhkq914dqv.feishu.cn/wiki/AeyTwVXcyism4Okm2CKclAHenjf)。你会用同一个回放与评测入口对照预训练权重和小规模微调，保留每帧输出、场景统计与复验记录。

## 仓库文件

- `annotations.csv` 是 55 张图的人工核对标注，按视频来源划分。帧号从 0 开始。框为原图像素下的 `x1,y1,x2,y2`。四列均空表示无球。
- `sources.json` 记录视频来源、署名与 SHA-256 校验值。公开视频为 CC BY 4.0，抽帧、缩放和标注为本练习所作改动。RT 视频沿用组内公开练习仓库的分享范围。
- `lab.py` 提供数据准备、CPU 微调、固定帧评测和逐帧回放。对命令加 `--help` 可查看参数。

## 跑通起始材料

已在 Ubuntu、Python 3.12、PyTorch 2.6 CPU 和 Ultralytics 8.3.232 上试跑。Ubuntu 或 WSL 需有 Python 虚拟环境支持和 curl。先克隆仓库，然后在仓库根目录运行。提交培训任务时，可以在自己的 Fork 中保留实验代码和结果。

```bash
git clone https://github.com/Ecrypted-Data/rt-basketball-detection-lab.git
cd rt-basketball-detection-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements.txt
python lab.py prepare
python lab.py replay --source data-src/rt.mp4 --limit 8 --output runs/replay-start.jsonl
```

`prepare` 会从清单中的链接下载原始视频和预训练权重，校验后生成 YOLO 格式数据。重复运行复用已校验的素材。准备完成后，可用 `python lab.py train` 做一次微调，权重在 `runs/lesson-fit/weights/best.pt`。回放默认使用初始权重。

`python lab.py eval --split val --output runs/val-baseline.jsonl` 会输出逐帧结果和分组汇总。用 `--weights`、`--imgsz`、`--conf` 改变要比较的条件，每次给结果一个不同的文件名。方案定下来后再选 `--split test`。具体提交和验收要求见飞书正文。

## 怎样读结果

每帧都有帧号和相对视频起点的时间。模型结果为空时，`detections` 是空列表。评测用原图坐标，匹配门槛固定为 IoU 0.50。标注触碰原图边界的有球帧归为 `edge`，其余完整框的短边折算到长边 640 后不超过 36 像素归为 `small`，再其余的为 `regular`。无球时为 `empty`。计数采用「匹配帧数 / 有球帧数」和「误报帧数 / 无球帧数」。

训练为 33 帧，来自 PLOS ONE 同一球馆的补充视频 S1、S2、S3、S6、S7 和 S9。验证为 RT 练习视频的 14 帧，测试为另一场比赛罚球视频的 8 帧。统计只代表这一组教学样例。
