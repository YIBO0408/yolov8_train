# -*- coding: utf-8 -*-
import argparse
import json
import os
import shutil
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

'''
统一图像格式
'''


def change_image_format(label_path, suffix='.jpg'):
    """
    统一当前文件夹下所有图像的格式为 RGB
    :param suffix: 图像文件后缀
    :param label_path: 当前文件路径
    :return:
    """
    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    files = []
    for extern in externs:
        files.extend(glob(os.path.join(label_path, f"*.{extern}")))
    for index, file in enumerate(tqdm(files)):
        name = ''.join(file.split('.')[:-1])
        file_suffix = file.split('.')[-1]
        if file_suffix != suffix.split('.')[-1]:
            new_name = name + suffix
            image = Image.open(file)
            image = image.convert("RGB")
            image.save(new_name)
            os.remove(file)


'''
读取所有json文件，获取所有的类别
'''


def get_all_class(file_list, label_path):
    """
    从json文件中获取当前数据的所有类别
    :param file_list:当前路径下的所有文件名
    :param label_path:当前文件路径
    :return:
    """
    classes = []
    for filename in tqdm(file_list):
        json_path = os.path.join(label_path, filename + '.json')
        json_file = json.load(open(json_path, "r", encoding="utf-8"))
        for item in json_file["shapes"]:
            label_class = item['label']
            if label_class not in classes:
                classes.append(label_class)
    print('read file done')
    return classes


'''
划分训练集、验证机、测试集
'''


def split_dataset(label_path, test_size=0.3, isUseTest=False, useNumpyShuffle=False):
    """
    将文件分为训练集，测试集和验证集
    :param useNumpyShuffle: 使用numpy方法分割数据集
    :param test_size: 分割测试集或验证集的比例
    :param isUseTest: 是否使用测试集，默认为False
    :param label_path: 当前文件路径
    :return: train_files, val_files, test_files, files
    """
    files = glob(label_path + "/*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]

    if useNumpyShuffle:
        file_length = len(files)
        index = np.arange(file_length)
        np.random.seed(32)
        np.random.shuffle(index)  # 随机划分

        test_files = None
        if isUseTest:
            trainval_files, test_files = np.array(files)[index[:int(file_length * (1 - test_size))]], np.array(files)[
                index[int(file_length * (1 - test_size)):]]
        else:
            trainval_files = files

        train_files = np.array(trainval_files)[index[:int(len(trainval_files) * (1 - test_size))]]
        val_files = np.array(trainval_files)[index[int(len(trainval_files) * (1 - test_size)):]]

    else:
        test_files = None
        if isUseTest:
            trainval_files, test_files = train_test_split(files, test_size=test_size, random_state=55)
        else:
            trainval_files = files

        train_files, val_files = train_test_split(trainval_files, test_size=test_size, random_state=55)

    return train_files, val_files, test_files, files


'''
生成yolov8的训练、验证、测试集的文件夹
'''


def create_save_file(ROOT_DIR, isUseTest=False):
    """
    生成YoloV8的训练、验证、测试集的文件夹结构
    :param isUseTest: 是否使用测试集
    :param ROOT_DIR: 数据集根目录
    :return: train_image, train_label, val_image, val_label, test_image, test_label
    """
    train_image = os.path.join(ROOT_DIR, 'images', 'train')
    train_label = os.path.join(ROOT_DIR, 'labels', 'train')
    os.makedirs(train_image, exist_ok=True)
    os.makedirs(train_label, exist_ok=True)

    val_image = os.path.join(ROOT_DIR, 'images', 'val')
    val_label = os.path.join(ROOT_DIR, 'labels', 'val')
    os.makedirs(val_image, exist_ok=True)
    os.makedirs(val_label, exist_ok=True)
    test_image = ""
    test_label = ""
    if isUseTest:
        test_image = os.path.join(ROOT_DIR, 'images', 'test')
        test_label = os.path.join(ROOT_DIR, 'labels', 'test')
        os.makedirs(test_image, exist_ok=True)
        os.makedirs(test_label, exist_ok=True)

    return train_image, train_label, val_image, val_label, test_image, test_label


'''
转换，根据图像大小，返回box框的中点和高宽信息
'''


def convert(size, box):
    # 宽
    dw = 1. / (size[0])
    # 高
    dh = 1. / (size[1])

    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    # 宽
    w = box[1] - box[0]
    # 高
    h = box[3] - box[2]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


'''
移动图像和标注文件到指定的训练集、验证集和测试集中
'''


def push_into_file(file, images, labels, ROOT_DIR, suffix='.jpg'):
    """
    最终生成在当前文件夹下的所有文件按image和label分别存在到训练集/验证集/测试集路径的文件夹下
    :param ROOT_DIR:
    :param file: 文件名列表
    :param images: 存放images的路径
    :param labels: 存放labels的路径
    :param suffix: 图像文件后缀
    :return:
    """
    for filename in tqdm(file):
        image_file = os.path.join(ROOT_DIR, filename + suffix)
        label_file = os.path.join(ROOT_DIR, filename + '.txt')
        # yolov8存放图像文件夹
        if not os.path.exists(os.path.join(images, filename + suffix)):
            try:
                shutil.move(image_file, images)
            except OSError:
                pass
        # yolov8存放标注文件夹
        if not os.path.exists(os.path.join(labels, filename + suffix)):
            try:
                shutil.move(label_file, labels)
            except OSError:
                pass


def json2txt(classes, modelType, txt_Name='allfiles', ROOT_DIR="", suffix='.jpg'):
    """
    将json文件转化为txt文件，并将json文件存放到指定文件夹
    :param classes: 类别名
    :param modelType: 模型类型，'det' 表示检测，'seg' 表示分割
    :param txt_Name: txt文件名，用来存放所有文件的路径
    :param ROOT_DIR: 根目录路径
    :param suffix: 图像文件后缀
    """
    # 创建存放转换后json文件的目录
    store_json = os.path.join(ROOT_DIR, 'json')
    if not os.path.exists(store_json):
        os.makedirs(store_json)

    # 获取数据集文件列表
    _, _, _, files = split_dataset(ROOT_DIR)

    # 创建用于记录文件路径的txt文件
    if not os.path.exists(os.path.join(ROOT_DIR, 'tmp')):
        os.makedirs(os.path.join(ROOT_DIR, 'tmp'))
    list_file = open(os.path.join(ROOT_DIR, 'tmp/%s.txt' % txt_Name), 'w')

    # 遍历所有json文件
    for json_file_ in tqdm(files):
        # 构建json文件路径
        json_filename = os.path.join(ROOT_DIR, json_file_ + ".json")
        # 构建图像文件路径
        image_path = os.path.join(ROOT_DIR, json_file_ + suffix)
        # 写入图像文件路径到txt文件中
        list_file.write('%s\n' % image_path)
        # 构建转换后的txt标签文件路径
        out_file = open('%s/%s.txt' % (ROOT_DIR, json_file_), 'w')
        # 加载标签json文件
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))

        # 标签转换：json转txt
        if modelType == "det":  # 检测模型类型
            if os.path.exists(image_path):
                height, width, _ = cv2.imread(image_path).shape
                for multi in json_file["shapes"]:
                    if len(multi["points"][0]) == 0:
                        out_file.write('')
                        continue
                    points = np.array(multi["points"])
                    xmin = min(points[:, 0]) if min(points[:, 0]) > 0 else 0
                    xmax = max(points[:, 0]) if max(points[:, 0]) > 0 else 0
                    ymin = min(points[:, 1]) if min(points[:, 1]) > 0 else 0
                    ymax = max(points[:, 1]) if max(points[:, 1]) > 0 else 0
                    label = multi["label"]
                    if xmax <= xmin or ymax <= ymin:
                        pass
                    else:
                        cls_id = classes.index(label)
                        b = (float(xmin), float(xmax), float(ymin), float(ymax))
                        bb = convert((width, height), b)
                        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        elif modelType == "seg":  # 分割模型类型
            if os.path.exists(image_path):
                height, width, _ = cv2.imread(image_path).shape
                for shape_dict in json_file["shapes"]:
                    label = shape_dict['label']
                    label_index = classes.index(label)
                    points = shape_dict["points"]
                    if len(points[0]) == 0:
                        out_file.write('')
                        continue
                    points_nor_list = [str(point[0] / width) + ' ' + str(point[1] / height) for point in points]
                    points_nor_str = ' '.join(points_nor_list)
                    label_str = str(label_index) + ' ' + points_nor_str + '\n'
                    out_file.writelines(label_str)

        # 将处理完的json文件移动到存放目录
        if not os.path.exists(os.path.join(store_json, json_file_ + '.json')):
            try:
                shutil.move(json_filename, store_json)
            except OSError:
                pass


'''
创建yaml文件
'''

import yaml


def create_yaml(classes, ROOT_DIR, isUseTest=False, dataYamlName=""):
    """
    生成用于训练YOLOv8模型的YAML文件
    :param classes: 类别名列表
    :param ROOT_DIR: 根目录路径
    :param isUseTest: 是否使用测试集，默认为False
    :param dataYamlName: YAML文件名
    """
    classes_dict = {index: item for index, item in enumerate(classes)}

    desired_caps = {
        'path': ROOT_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': classes_dict
    }
    if isUseTest:
        desired_caps['test'] = 'images/test'

    yamlpath = os.path.join(ROOT_DIR, f"{dataYamlName}.yaml")

    # 写入到yaml文件
    with open(yamlpath, "w+", encoding="utf-8") as f:
        yaml.dump(desired_caps, f, default_flow_style=False)


def ChangeToYolo(ROOT_DIR="", suffix='.bmp', modelType="det", classes="", test_size=0.1, isUseTest=False,
                 useNumpyShuffle=False, auto_genClasses=True, dataYamlName=""):
    """
    将数据转换为YOLOv8模型所需的标准格式
    :param ROOT_DIR: 数据根目录路径
    :param suffix: 图像文件后缀名
    :param modelType: 模型类型，"det"表示目标检测，"seg"表示语义分割
    :param classes: 类别名列表
    :param test_size: 分割测试集或验证集的比例
    :param isUseTest: 是否使用测试集
    :param useNumpyShuffle: 是否使用numpy方法分割数据集
    :param auto_genClasses: 是否自动从数据集中获取类别数
    :param dataYamlName: YAML文件名
    """
    # step1: 统一图像格式
    change_image_format(ROOT_DIR)
    print(f"统一图片格式为{suffix}完毕！")

    # 复制一份原始数据集至dataset目录
    dataset_name = ROOT_DIR.split("/")[-1]
    dataset_dir = str(Path(__file__).absolute().parent.parent.parent / "dataset" / dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    if os.listdir(dataset_dir):
        shutil.rmtree(dataset_dir, ignore_errors=True)
    shutil.copytree(ROOT_DIR, dataset_dir, dirs_exist_ok=True)

    # step2: 根据json文件划分训练集、验证集、测试集
    train_files, val_files, test_files, files = split_dataset(dataset_dir, test_size=test_size, isUseTest=isUseTest,
                                                              useNumpyShuffle=useNumpyShuffle)
    print("划分数据集完毕！")

    # step3: 是否自动从数据集中获取类别数
    if auto_genClasses:
        classes = get_all_class(files, dataset_dir)
        print(f"从数据集中获取类别数为{classes}")

    # step4: 将json文件转化为txt文件，并将json文件存放到指定文件夹
    json2txt(classes, modelType, txt_Name='allfiles', ROOT_DIR=dataset_dir, suffix=suffix)
    print("标签文件转化完毕！")

    # step5: 创建yolov8训练所需的yaml文件
    create_yaml(classes, dataset_dir, isUseTest=isUseTest, dataYamlName=dataYamlName)
    print("构建训练所需yaml文件完毕！")

    # step6: 生成yolov8的训练、验证、测试集的文件夹
    train_image_dir, train_label_dir, val_image_dir, val_label_dir, test_image_dir, test_label_dir = create_save_file(
        dataset_dir, isUseTest)

    # step7: 将所有图像和标注文件，移动到对应的训练集、验证集、测试集
    push_into_file(train_files, train_image_dir, train_label_dir, ROOT_DIR=dataset_dir, suffix=suffix)
    push_into_file(val_files, val_image_dir, val_label_dir, ROOT_DIR=dataset_dir, suffix=suffix)
    if isUseTest and test_files is not None:
        push_into_file(test_files, test_image_dir, test_label_dir, ROOT_DIR=dataset_dir, suffix=suffix)

    print("生成yolov8的训练、验证、测试集的文件夹完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to YOLO format")
    parser.add_argument("--ROOT_DIR", type=str, required=True,
                        help="Path to the directory containing images and JSON labels")
    parser.add_argument("--suffix", type=str, default='.jpg', help="Image file suffix")
    parser.add_argument("--modelType", type=str, default="det", choices=["det", "seg"],
                        help="Model type: 'det' for object detection, 'seg' for semantic segmentation")
    parser.add_argument("--dataYamlName", type=str, default="yolov8_ocr", help="Name of the data YAML file")

    args = parser.parse_args()

    ChangeToYolo(ROOT_DIR=args.ROOT_DIR,
                 suffix=args.suffix,
                 modelType=args.modelType,
                 dataYamlName=args.dataYamlName)
