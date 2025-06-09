# 模型文件说明

本文件夹包含项目中使用的各种YOLO系列模型权重文件。这些模型用于足球比赛视频中的目标检测与跟踪任务。

## 文件列表

- **model_yolov12n.pt**: YOLOv12的nano版本模型权重，是本项目的主要模型
- **model_yolov8n.pt**: YOLOv8的nano版本模型权重
- **model_yolov8n_original.pt**: YOLOv8 nano的原始预训练权重
- **model_yolo11n.pt**: YOLOv11的nano版本模型权重
- **model_rtdetr-x.pt**: RT-DETR (Real-Time Detection Transformer)的X版本模型权重
- **model_yolov10x.pt**: YOLOv10的X版本模型权重（较大模型）

## 使用说明

这些模型可以通过ultralytics库加载：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("models/model_yolov12n.pt")

# 使用模型进行推理
results = model.predict(image)  # 单张图像推理
# 或
results = model.track(frame, persist=True)  # 视频跟踪
```

## 模型选择指南

- 对于实时检测任务，推荐使用nano版本的模型（yolov12n, yolov8n）
- 对于更高精度的检测任务，可使用更大的模型（yolov10x, rtdetr-x）
- 本项目主要使用YOLOv12 nano模型，它在速度和精度之间取得了较好的平衡 