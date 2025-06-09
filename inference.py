"""
这个文件实现了模型推理功能。
主要用于对测试视频进行处理，通过加载训练好的YOLOv8模型，
对视频帧进行目标检测和跟踪，生成符合比赛评测格式的结果文件，
并最终将结果打包成zip文件提交。
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
            results = model.track(frame, persist=True, verbose=False)
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
    model = YOLO(r"/home/aistudio/work/mots_code/codeing/runs/detect/yolov8n_train5/weights/best.pt", verbose=False)
    base_image_folder = r"/home/aistudio/work/mots_code/test"
    output_dir = os.path.join("IKCEST-test", "IKCEST")
    os.makedirs(output_dir, exist_ok=True)

    video_folders = [f.path for f in os.scandir(base_image_folder) if f.is_dir()]

    for video_folder in video_folders:
        video_name = os.path.basename(video_folder)
        image_folder = os.path.join(video_folder, "img1")
        if os.path.exists(image_folder):
            create_results_txt(video_name, output_dir, model, image_folder)

    zip_results("IKCEST-test", "results.zip")
    print("推理完成，结果已保存并压缩为results.zip")

if __name__ == "__main__":
    main()