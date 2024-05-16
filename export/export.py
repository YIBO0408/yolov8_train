from pathlib import Path

from ultralytics import YOLO
import argparse


def export(args):
    model = YOLO(args.model_path)
    model.export(
        format=args.export_format,  # 导出模型的目标格式。
        # imgsz=768,  # 模型输入所需的图像尺寸。对于正方形图像，可以是一个整数，或者是一个元组 (height, width) 了解具体尺寸。
        keras=False,  # 启用导出为 Keras 格式的TensorFlow SavedModel ，提供与TensorFlow serving 和 API 的兼容性。
        optimize=False,  # 在导出到TorchScript 时，应用针对移动设备的优化，可能会减小模型大小并提高性能。
        half=False,  # 启用 FP16（半精度）量化，在支持的硬件上减小模型大小并可能加快推理速度。
        int8=False,  # 激活 INT8 量化，进一步压缩模型并加快推理速度，同时将精度损失降至最低，主要用于边缘设备。
        dynamic=True,  # 允许ONNX 和TensorRT 导出动态输入尺寸，提高了处理不同图像尺寸的灵活性。
        simplify=False,  # 简化了ONNX 导出的模型图，可能会提高性能和兼容性。
        opset=None,  # 指定ONNX opset 版本，以便与不同的ONNX 解析器和运行时兼容。如果未设置，则使用最新的支持版本。
        workspace=4,  # 以 GB 为单位设置最大工作区大小，用于TensorRT 优化，平衡内存使用和性能。
        nms=False,  # 在CoreML 导出中添加非最大值抑制 (NMS)，这对精确高效的检测后处理至关重要。
        batch=1
    )
    out_path = Path(args.model_path).parent.resolve()
    with open(out_path / "class_names_list.txt", "w") as f:
        for i in model.names:
            f.write(f"{model.names[i]}\n")
    f.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help="训练好后的best.pt的绝对路径", type=str)
    parser.add_argument("-e", "--export_format", help="导出格式", default="onnx", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    export(get_args())
