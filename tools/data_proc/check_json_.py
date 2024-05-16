import os
import json

# json目录
fileDir = '/home/yibo/yolov8_train/综合样本_20240418091311'


# 使用os.listdir()来列出目录中的所有文件
file_names = os.listdir(fileDir)

# 获取该目录下的.json 文件名称
json_files = [file for file in file_names if file.endswith('.json')]

w = []
y = []
# 遍历指定目录下的 .json文件名称
for json_file in json_files:

    # 文件名称 去掉 .json
    fileName = json_file.replace('.json', '')
    print(fileName)
    # 读取json文件内容,返回字典格式
    with open(f'{fileDir}/{fileName}.json', 'r', encoding='utf8') as fp:
        json_data = json.load(fp)

    # 创建需要返回的字典  my_dict{'分类名称:[point xy坐标]'}
    shaixuan = json_data['shapes']
    for element in shaixuan:
        if 'shape_type' in element and element['shape_type'] != 'polygon':
            print(json_file)
            w += [json_file]
            y += [element['shape_type']]
            break
    else:
        print("All elements have shape_type 'polygon'")
print(w)
print(y)