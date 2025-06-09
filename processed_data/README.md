# 处理后的数据文件夹

该文件夹用于存储经过预处理后的数据，用于模型训练和验证。

## 预期内容

这个文件夹应包含：

- 经过`data_processor.py`处理后的YOLO格式数据
- 训练集和验证集的图像和标签
- 相关的配置文件（如data.yaml）

## 数据结构

处理后的数据通常按以下结构组织：

```
processed_data/
├── train/
│   ├── images/
│   │   ├── video1-000001.jpg
│   │   ├── video2-000005.jpg
│   │   └── ...
│   └── labels/
│       ├── video1-000001.txt
│       ├── video2-000005.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

其中：
- `images`文件夹包含处理后的图像
- `labels`文件夹包含YOLO格式的标注文件
- `data.yaml`包含数据集配置信息

## 与其他文件的关系

- 原始数据来源于`raw_data`文件夹
- 数据处理过程由`data_processor.py`脚本实现
- 处理后的数据直接用于`train_model.py`中的模型训练 