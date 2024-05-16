import cv2
import numpy as np
from ultralytics import YOLO


def cv_show(name: str, img):
    cv2.namedWindow(name, 0)
    cv2.imshow(name, img)
    cv2.waitKey(100)
    # cv2.destroyAllWindows()
    return


class ClsPredict():
    # 预训练权重，训练权重
    def __init__(self, official_model, custom_model) -> None:
        self.model = YOLO(official_model)  # load an official model
        self.model = YOLO(custom_model)  # load a custom model

    # 单张预测
    def predict(self, img):
        results = self.model(img)

        # 各类别名称
        names = results[0].names
        # 各类别置信度
        confs = results[0].probs.data.cpu().numpy()
        # 置信度最高的索引
        max_index = np.argmax(confs)

        state = names[max_index]
        score = confs[max_index]

        return state + str(score)


if __name__ == "__main__":
    # 预训练权重地址（与训练时使用的是同一个）
    official_model = '/home/yibo/yolov8_train/models/yolov8s-cls.pt'
    # 训练的得到的权重的地址
    custom_model = '/home/yibo/yolov8_train/cls_data/train_ball/s/weights/best.pt'

    clspredic = ClsPredict(official_model, custom_model)

    img = cv2.imread("/home/yibo/yolov8_train/cls_data/data/test/baseball/47dbef3d-b281-472f-8c65-26d523d460e3_jpg.rf.a8c74fac386facb5fbff905dbf77b175.jpg")
    results = clspredic.predict(img)
    print(results)