import os

directory = "/home/yibo/yolov8_train/综合样本_rgb"

for filename in os.listdir(directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_filepath = os.path.join(directory, json_filename)
        if not os.path.exists(json_filepath):
            img_filepath = os.path.join(directory, filename)
            os.remove(img_filepath)
            print(f"已删除 {filename}，因为没有对应的 JSON 文件")
