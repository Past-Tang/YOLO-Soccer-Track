"""
这个文件实现了足球比赛数据集的预处理功能。
主要功能包括：
1. 将原始数据集转换为YOLO格式（包括标签转换和图像处理）
2. 数据集分割（训练集和验证集）
3. 多线程并行处理加速数据转换
4. 处理足球比赛视频中的不同目标类别（球员、裁判、足球）
该文件提供了完整的数据处理流程，从原始视频帧和标注文件到可直接用于YOLO模型训练的格式。
"""
import os
import glob
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import shutil


def format_number(number, length=6):
    return str(str(number).zfill(length)) + ".jpg"


def one_type(seqtext):
    data_dict = {}
    for item in seqtext.split("\n")[11:][:-1]:
        key, value = item.split('= ')
        data_dict[key] = value
    modified_dict = {}
    for key, value in data_dict.items():
        if 'player' in value or 'goalkeeper' in value:
            modified_dict[key] = 'player'
        elif 'referee' in value:
            modified_dict[key] = 'referee'
        elif 'ball' in value:
            modified_dict[key] = 'ball'
    return modified_dict

def copy_file(src, dst):
    shutil.copy2(src, dst)

def convert_to_yolo_format(bb_left, bb_top, width, height, image_width, image_height, class_index):
    center_x = (bb_left + width / 2) / image_width
    center_y = (bb_top + height / 2) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return f"{class_index} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"




def process_single_directory(directory, output_dir, dataset):
    with open(os.path.join(directory, "gameinfo.ini"), "r") as f:
        type_dict = one_type(f.read())

    with open(os.path.join(directory, "gt", "gt.txt"), "r") as f:
        lines = f.read().strip().split("\n")
        data = [list(map(int, line.split(","))) for line in lines]
        df = pd.DataFrame(data,columns=['frame_id', 'trackletID', 'bb_left', 'bb_top', 'width', 'height', 'mark', 'col8','col9', 'col10'])
        grouped = df.groupby('frame_id', group_keys=False).apply(lambda x: x.values.tolist()).tolist()
    for j in grouped:
        yolo_text = ''
        img_src = os.path.join(directory, "img1", format_number(j[0][0]))
        for k in j:
            bb_left, bb_top, width, height = k[2], k[3], k[4], k[5]
            image_width, image_height = 1920, 1080
            value = type_dict.get(f'trackletID_{k[1]}')
            class_index = {'player': 0, 'referee': 1, 'ball': 2}.get(value, -1)
            # if class_index == 2:
            #     continue
            if class_index != -1:
                yolo_label = convert_to_yolo_format(bb_left, bb_top, width, height, image_width, image_height,class_index)
                yolo_text += yolo_label + "\n"
        img_name = f"{os.path.basename(directory)}-{format_number(j[0][0])}"
        img_dst = os.path.join(output_dir, dataset, 'images', img_name)
        txt_path = os.path.join(output_dir, dataset, 'labels', img_name.replace(".jpg", ".txt"))
        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(yolo_text)
        copy_file(img_src, img_dst)


def count_images_in_directory(directory):
    return sum(len(files) for _, _, files in os.walk(os.path.join(directory, "img1")))


def process_dataset(input_dir, output_dir, train_ratio=0.8):
    os.makedirs(os.path.join(output_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid', 'labels'), exist_ok=True)

    directories = glob.glob(os.path.join(input_dir, '*'))

    # 计算需要处理的总图像数
    total_images = sum(count_images_in_directory(directory) for directory in directories)
    print(f"需要处理的图像总数：{total_images}")

    random.shuffle(directories)
    split_index = int(len(directories) * train_ratio)
    train_dirs = directories[:split_index]
    val_dirs = directories[split_index:]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        train_futures = [executor.submit(process_single_directory, directory, output_dir, 'train') for directory in
                         train_dirs]
        val_futures = [executor.submit(process_single_directory, directory, output_dir, 'valid') for directory in
                       val_dirs]

        for future in tqdm.tqdm(as_completed(train_futures + val_futures), total=len(directories), desc="处理数据集"):
            try:
                future.result()
            except Exception as exc:
                print(f'处理目录时生成了一个异常: {exc}')

    train_images = sum(len(files) for _, _, files in os.walk(os.path.join(output_dir, 'train', 'images')))
    val_images = sum(len(files) for _, _, files in os.walk(os.path.join(output_dir, 'valid', 'images')))

    print(f"处理完成。共{train_images + val_images}张图片，其中训练集{train_images}张，验证集{val_images}张。")


if __name__ == '__main__':
    input_directory = r"data/train"
    output_directory = r"yolo"
    train_ratio = 0.8  # 80% 用于训练，20% 用于验证

    process_dataset(input_directory, output_directory, train_ratio)