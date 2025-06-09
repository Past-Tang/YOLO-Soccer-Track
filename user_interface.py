"""
这个文件实现了基于YOLOv12的足球比赛实时跟踪系统的用户界面。
通过Gradio框架构建了交互式Web界面，提供以下功能：
1. 上传视频进行目标检测与跟踪
2. 摄像头实时检测（视频模式和单帧图像模式）
3. 显示检测结果和统计信息
4. 可调节的检测置信度阈值
该界面支持实时可视化目标检测结果，包括球员、裁判、足球等目标的识别与跟踪。
"""
import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from collections import defaultdict
import time

# Load YOLO model
model = YOLO("/home/tang/bishe/codeing/runs/detect/train2/weights/best.pt")

# Global variable to store model info
model_info = {
    "name": "基于YOLOv12的足球比赛实时跟踪系统实现",
    "classes": model.names,
    "version": "YOLOv12"
}

# Process uploaded video
def process_video(video_path, conf_threshold=0.25, progress=gr.Progress()):
    if video_path is None:
        return None, "请上传视频文件"
        
    # Create temp file for output video
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    output_path = temp_output.name
    temp_output.close()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Detection statistics
    detection_stats = defaultdict(int)
    frame_count = 0
    
    # Track history
    track_history = defaultdict(lambda: [])
    
    progress(0, desc="初始化处理...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Update progress
        frame_count += 1
        progress(frame_count / total_frames, desc=f"处理中... {frame_count}/{total_frames} 帧")
        
        # Run YOLO tracking
        results = model.track(frame, persist=True, conf=conf_threshold)
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Update statistics
            for cls in boxes.cls.cpu().numpy():
                cls_id = int(cls)
                class_name = model.names[cls_id]
                detection_stats[class_name] += 1
            
            # Plot results on frame
            annotated_frame = results[0].plot(conf=False, line_width=1)
            out.write(annotated_frame)
        else:
            # If no detection, write original frame
            out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    # Format statistics
    stats_text = "检测统计:\n"
    for class_name, count in detection_stats.items():
        stats_text += f"- {class_name}: {count} 次检测\n"
    
    return output_path, stats_text

# Process a static image from webcam or upload
def process_image(image, conf_threshold=0.25):
    if image is None:
        return None, "无图像输入"
    
    # Convert to OpenCV format if needed
    if isinstance(image, np.ndarray):
        frame = image.copy()
        if len(frame.shape) == 3 and frame.shape[2] == 4:  # If has alpha channel, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    else:
        return None, "图像格式不支持"
    
    # Run YOLO detection
    results = model.predict(frame, conf=conf_threshold)
    
    # Annotate the frame with the detection results
    annotated_frame = frame.copy()  # Copy the frame to ensure original remains intact
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            confidence = float(box.conf.item())
            
            # Draw bounding box and label
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get coordinates of bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{class_name} {confidence:.2f}", 
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Generate detection information
    info_text = "检测结果:\n"
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            confidence = float(box.conf.item())
            info_text += f"- 对象 {i+1}: {class_name} (置信度: {confidence:.2f})\n"
    else:
        info_text += "未检测到任何对象"
    
    return annotated_frame, info_text
# Live webcam stream with continuous processing
def webcam_live(conf_threshold=0.15):
    return gr.update(visible=True), gr.update(visible=False)

# Process webcam video
def webcam_stream(conf_threshold=0.25, duration=5, progress=gr.Progress()):
    cap = cv2.VideoCapture(0)  # Open default webcam
    
    if not cap.isOpened():
        return None, "无法访问摄像头，请确认权限和连接"
    
    # Create temp file for output video
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    output_path = temp_output.name
    temp_output.close()
    
    # Get webcam parameters
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0  # Fixed framerate
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Counter, control recording duration
    max_frames = int(fps * duration)
    detection_stats = defaultdict(int)
    
    progress(0, desc="准备录制...")
    
    start_time = time.time()
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break
        
        # Update progress
        frame_count += 1
        elapsed = time.time() - start_time
        remaining = max(0, duration - elapsed)
        progress(frame_count / max_frames, desc=f"录制中... 剩余 {remaining:.1f} 秒")
        
        # Run YOLO detection
        results = model.predict(frame, conf=conf_threshold)
        
        # Update statistics
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for cls in boxes.cls.cpu().numpy():
                cls_id = int(cls)
                class_name = model.names[cls_id]
                detection_stats[class_name] += 1
        
        annotated_frame = results[0].plot(conf=False, line_width=2)
        
        # Write video frame
        out.write(annotated_frame)
    
    # Release resources
    cap.release()
    out.release()
    
    # Format statistics
    stats_text = "检测统计:\n"
    for class_name, count in detection_stats.items():
        stats_text += f"- {class_name}: {count} 次检测\n"
    
    if not detection_stats:
        stats_text += "未检测到任何对象"
    
    return output_path, stats_text

# Create Gradio interface
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 基于YOLOv12的足球比赛实时跟踪系统实现")
        
        # Model information
        with gr.Accordion("模型信息", open=False):
            gr.Markdown(f"""
            - **版本**: {model_info['version']}
            - **支持的目标类别**: {', '.join(model_info['classes'].values())}
            """)
        
        # Main tabs
        with gr.Tabs() as tabs:
            # Video Upload Tab
            with gr.TabItem("视频检测") as video_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="上传视频", format="mp4")
                        with gr.Row():
                            conf_slider = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                                label="置信度阈值", 
                                info="检测置信度阈值，值越低检测到的目标越多，但可能增加误检"
                            )
                        submit_btn = gr.Button("开始处理", variant="primary")
                        video_stats = gr.Textbox(label="检测统计", lines=6)
                        
                    with gr.Column(scale=2):
                        video_output = gr.Video(label="处理结果")
                
                submit_btn.click(
                    fn=process_video,
                    inputs=[video_input, conf_slider],
                    outputs=[video_output, video_stats]
                )
            
            # Webcam Tab
            with gr.TabItem("摄像头检测") as webcam_tab:
                with gr.Tabs() as webcam_modes:
                    # Webcam Video Mode
                    with gr.TabItem("摄像头视频") as webcam_video_mode:
                        with gr.Row():
                            with gr.Column(scale=1):
                                webcam_conf_slider = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                                    label="置信度阈值"
                                )
                                duration_slider = gr.Slider(
                                    minimum=1, maximum=10, value=5, step=1,
                                    label="录制时长（秒）"
                                )
                                webcam_btn = gr.Button("启动摄像头录制", variant="primary")
                                webcam_stats = gr.Textbox(label="检测统计", lines=6)
                                
                            with gr.Column(scale=2):
                                webcam_output = gr.Video(label="摄像头检测结果")
                        
                        webcam_btn.click(
                            fn=webcam_stream,
                            inputs=[webcam_conf_slider, duration_slider],
                            outputs=[webcam_output, webcam_stats]
                        )
                    
                    # Webcam Image Mode
                    with gr.TabItem("单张图像检测") as webcam_image_mode:
                        with gr.Row():
                            with gr.Column(scale=1):
                                webcam_image_conf_slider = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                                    label="置信度阈值"
                                )
                                webcam_image_input = gr.Image(
                                    label="拍照或上传图片",
                                    sources=["webcam", "upload"],
                                    type="numpy"
                                )
                                image_info = gr.Textbox(label="检测信息", lines=6)
                                
                            with gr.Column(scale=2):
                                webcam_image_output = gr.Image(label="检测结果")
                        
                        webcam_image_input.change(
                            fn=process_image,
                            inputs=[webcam_image_input, webcam_image_conf_slider],
                            outputs=[webcam_image_output, image_info]
                        )
        
        # Footer
        gr.Markdown("""
        ### 使用说明
        - **视频检测**: 上传视频文件，设置置信度阈值，点击"开始处理"按钮
        - **摄像头视频**: 设置参数后点击"启动摄像头录制"，将录制并处理指定时长的视频
        - **单张图像检测**: 拍照或上传图片进行实时检测
        """)
    
    return demo

# Main program
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False, server_name="0.0.0.0")