"""
这个文件是IKCEST比赛的初始推理脚本。
主要功能包括：
1. 加载训练好的YOLO模型进行目标检测和跟踪
2. 处理测试视频序列中的每一帧图像
3. 根据比赛要求生成标准格式的结果文本文件
4. 将结果打包成zip文件用于提交评测
该脚本是足球比赛视频目标跟踪任务的测试阶段使用。
"""
import os
import glob
import cv2
from ultralytics import YOLO
import zipfile

def create_results_txt(video_name, output_dir, model, image_folder):
    output_file = os.path.join(output_dir, f"{video_name}.txt")
    with open(output_file, 'w') as f:
        image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
        for frame_id, image_file in enumerate(image_files):
            frame = cv2.imread(image_file)
            results = model.track(frame, persist=True, verbose=False,device=[1])
            for result in results:
                for box in result.boxes:
                    track_id = int(box.id) if box.id is not None else -1
                    bbox_left = box.xyxy[0][0].item()
                    bbox_top = box.xyxy[0][1].item()
                    bbox_width = box.xyxy[0][2].item() - bbox_left
                    bbox_height = box.xyxy[0][3].item() - bbox_top
                    score = box.conf.item()
                    cls_id = 1  # 评测时统一设置为1
                    f.write(f"{frame_id+1},{track_id},{bbox_left:.2f},{bbox_top:.2f},{bbox_width:.2f},{bbox_height:.2f},{score:.2f},{cls_id},-1,-1\n")

def zip_results(base_dir, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(base_dir))
                zipf.write(file_path, arcname=arcname)

def main():
    #模型地址
    model = YOLO(r"model.pt", verbose=False)
    #test地址
    base_image_folder = r"../test"
    output_dir = os.path.join("IKCEST-test", "IKCEST")
    os.makedirs(output_dir, exist_ok=True)

    video_folders = [f.path for f in os.scandir(base_image_folder) if f.is_dir()]
    print("请耐心等待...")
    for video_folder in video_folders:
        video_name = os.path.basename(video_folder)
        image_folder = os.path.join(video_folder, "img1")
        if os.path.exists(image_folder):
            create_results_txt(video_name, output_dir, model, image_folder)

    zip_results("IKCEST-test", "results.zip")
    print("推理完成，结果已保存并压缩为results.zip")

if __name__ == "__main__":
    main()