"""
这个文件用于训练YOLOv12模型进行足球比赛目标检测和跟踪。
主要功能：
1. 加载预训练的YOLOv12模型
2. 配置详细的训练参数（包括学习率、数据增强策略、批量大小等）
3. 训练模型用于足球比赛场景中的目标检测（球员、裁判、足球）
4. 支持多GPU并行训练加速
该文件可以直接执行，用于训练自定义数据集上的YOLOv12模型。
"""
from ultralytics import YOLO

# 加载模型
model = YOLO("/home/tang/bishe/codeing/yolov12n.pt")  # 使用配置加载预训练模型
# 定义训练参数
train_params = {
    "data": "/home/aistudio/work/mots_code/codeing/yolo/data.yaml",  # 数据集配置文件的路径
    "resume":True,
    "epochs": 10,  # 训练的轮数
    "batch": 13,  # 批处理大小
    "imgsz": 1024,  # 图像尺寸
    "save": True,  # 是否保存检查点和最终模型
    "save_period": -1,  # 保存检查点的频率（-1表示禁用）
    "cache": False,  # 是否将数据集图像缓存到内存中
    "device": [0],  # 用于训练的设备（多GPU示例）
    "workers": 8,  # 数据加载的工作线程数
    "name": "yolov8n_train",  # 训练运行的名称
    "pretrained": True,  # 是否从预训练模型开始训练
    "optimizer": 'SGD',  # 优化器选择
    "resume": False,  # 是否从最后的检查点继续训练
    "freeze": None,  # 冻结模型的前N层
    "lr0": 0.01,  # 初始学习率
    "lrf": 0.01,  # 最终学习率为lr0的比例
    "nbs": 64,  # 损失归一化的名义批处理大小
    "overlap_mask": True,  # 训练期间是否重叠分割掩码
    "mask_ratio": 4,  # 分割掩码的降采样比例
    "dropout": 0.0,  # 分类任务中的随机丢弃率
    "val": True,  # 训练期间是否启用验证
    "plots": True,  # 生成并保存训练图
    "hsv_h": 0.015,  # 色调增强
    "hsv_s": 0.7,  # 饱和度增强
    "hsv_v": 0.4,  # 亮度增强
    "degrees": 0.0,  # 旋转增强
    "translate": 0.1,  # 平移增强
    "scale": 0.5,  # 缩放增强
    "shear": 0.0,  # 剪切增强
    "perspective": 0.0,  # 透视增强
    "flipud": 0.0,  # 垂直翻转增强
    "fliplr": 0.5,  # 水平翻转增强
    "bgr": 0.0,  # RGB到BGR通道翻转增强
    "mosaic": 1.0,  # 马赛克增强
    "mixup": 0.0,  # Mixup增强
    "copy_paste": 0.0,  # 复制粘贴增强
    "auto_augment": "randaugment",  # 自动增强策略
    "erasing": 0.4,  # 随机擦除增强
    "crop_fraction": 1.0,  # 分类图像的裁剪比例
}

# 使用指定参数训练模型
results = model.train(epochs=50,data="/home/tang/bishe/codeing/yolo/data.yaml",batch= 256,device=[0,1,2,3,4,5,6,7],imgsz=1024,patience=5)