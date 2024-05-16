import cv2
import os

original_directory = "/home/yibo/yolov8_train/综合样本_20240418091311"
new_directory = "/home/yibo/yolov8_train/综合样本_rgb"

if not os.path.exists(new_directory):
    os.makedirs(new_directory)

for filename in os.listdir(original_directory):
    original_filepath = os.path.join(original_directory, filename)
    if os.path.isfile(original_filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        img = cv2.imread(original_filepath, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            new_filepath = os.path.join(new_directory, filename)
            cv2.imwrite(new_filepath, img_rgb)
            print(f"已将 {filename} 转换为 RGB 模式并保存到 {new_directory}")
        else:
            new_filepath = os.path.join(new_directory, filename)
            os.system(f'cp "{original_filepath}" "{new_filepath}"')
            print(f"{filename} 不是 RGBA 图像，无需转换。已复制 {filename} 到 {new_directory}")
