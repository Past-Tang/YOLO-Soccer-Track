# IKCEST 比赛初始文件

这个文件夹包含IKCEST足球比赛目标跟踪挑战赛的初始文件和参考实现。

## 文件说明

- **model.pt**: 训练模型权重文件
- **Inference.py**: 推理脚本
- **requirements.txt**: 环境依赖列表

## 与项目关系

这个文件夹中的内容作为本项目的起点，我们在此基础上进行了以下改进：

1. 重构了推理脚本，优化了处理流程（见项目根目录的`inference.py`）
2. 添加了数据预处理功能（`data_processor.py`）
3. 实现了模型训练流程（`train_model.py`）
4. 开发了用户交互界面（`user_interface.py`）

## 使用参考

初始推理脚本的使用方式：

```bash
# 切换到IKCEST_initial目录
cd IKCEST_initial

# 运行推理脚本
python Inference.py
```

该脚本将处理测试视频并输出结果文件，用于比赛提交。 