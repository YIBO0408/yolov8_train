import argparse

import torch
from ultralytics import YOLO


def train(args):
    model = YOLO(f'yolov8{args.model_type}.yaml').load(f'yolov8{args.model_type}.pt')
    device = "cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu"
    model.train(
        project=args.project_name,  # 保存训练结果的项目目录名称。允许有组织地存储不同的实验。
        name=args.model_type,  # 训练运行的名称。用于在项目文件夹内创建一个子目录，用于存储训练日志和输出结果。
        model=model,  # 指定用于训练的模型文件。接受指向 .pt 预训练模型或 .yaml 配置文件。对于定义模型结构或初始化权重至关重要。
        data=args.yaml_path,  # 数据集配置文件的路径（例如 coco128.yaml).该文件包含特定于数据集的参数，包括训练数据和验证数据的路径、类名和类数。
        epochs=args.epoch,  # 训练历元总数。每个历元代表对整个数据集进行一次完整的训练。调整该值会影响训练时间和模型性能。
        # time=None,  # 最长训练时间（小时）。如果设置了该值，则会覆盖 epochs 参数，允许训练在指定的持续时间后自动停止。对于时间有限的训练场景非常有用。
        patience=args.epoch,  # 在验证指标没有改善的情况下，提前停止训练所需的历元数。当性能趋于平稳时停止训练，有助于防止过度拟合。
        batch=2,  # 训练的批量大小，表示在更新模型内部参数之前要处理多少张图像。自动批处理 (batch=-1)会根据 GPU 内存可用性动态调整批处理大小。
        imgsz=args.img_size,  # 用于训练的目标图像尺寸。所有图像在输入模型前都会被调整到这一尺寸。影响模型精度和计算复杂度。
        save=True,  # 可保存训练检查点和最终模型权重。这对恢复训练或模型部署非常有用。
        save_period=-1,  # 保存模型检查点的频率，以 epochs 为单位。值为-1 时将禁用此功能。该功能适用于在长时间训练过程中保存临时模型。
        cache=False,  # 在内存中缓存数据集图像 (True/ram）、磁盘 (disk），或禁用它 (False).通过减少磁盘 I/O 提高训练速度，但代价是增加内存使用量。
        device=device,  # 指定用于训练的计算设备：单个 GPU (device=0）、多个 GPU (device=0,1)、CPU (device=cpu)，或苹果芯片的 MPS (device=mps).
        workers=4,  # 加载数据的工作线程数（每 RANK 多 GPU 训练）。影响数据预处理和输入模型的速度，尤其适用于多 GPU 设置。
        exist_ok=False,  # 如果为 True，则允许覆盖现有的项目/名称目录。这对迭代实验非常有用，无需手动清除之前的输出。
        pretrained=True,  # 决定是否从预处理模型开始训练。可以是布尔值，也可以是加载权重的特定模型的字符串路径。提高训练效率和模型性能。
        optimizer='auto',  # 为培训选择优化器。选项包括 SGD, Adam, AdamW, NAdam, RAdam, RMSProp 等，或 auto 用于根据模型配置进行自动选择。影响收敛速度和稳定性
        verbose=False,  # 在训练过程中启用冗长输出，提供详细日志和进度更新。有助于调试和密切监控培训过程。
        seed=0,  # 为训练设置随机种子，确保在相同配置下运行的结果具有可重复性。
        deterministic=True,  # 强制使用确定性算法，确保可重复性，但由于对非确定性算法的限制，可能会影响性能和速度。
        single_cls=False,  # 在训练过程中将多类数据集中的所有类别视为单一类别。适用于二元分类任务，或侧重于对象的存在而非分类。
        rect=False,  # 可进行矩形训练，优化批次组成以减少填充。这可以提高效率和速度，但可能会影响模型的准确性。
        cos_lr=False,  # 利用余弦学习率调度器，根据历时的余弦曲线调整学习率。这有助于管理学习率，实现更好的收敛。
        close_mosaic=10,  # 在训练完成前禁用最后 N 个历元的马赛克数据增强以稳定训练。设置为 0 则禁用此功能。
        resume=args.resume,  # 从上次保存的检查点恢复训练。自动加载模型权重、优化器状态和历时计数，无缝继续训练。
        amp=True,  # 启用自动混合精度 (AMP) 训练，可减少内存使用量并加快训练速度，同时将对精度的影响降至最低。
        fraction=1.0,  # 指定用于训练的数据集的部分。允许在完整数据集的子集上进行训练，这对实验或资源有限的情况非常有用。
        profile=False,  # 在训练过程中，可对ONNX 和TensorRT 速度进行剖析，有助于优化模型部署。
        freeze=None,  # 冻结模型的前 N 层或按索引指定的层，从而减少可训练参数的数量。这对微调或迁移学习非常有用。
        lr0=0.01,  # 初始学习率（即 SGD=1E-2, Adam=1E-3) .调整这个值对优化过程至关重要，会影响模型权重的更新速度。
        lrf=0.01,  # 最终学习率占初始学习率的百分比 = (lr0 * lrf)，与调度程序结合使用，随着时间的推移调整学习率。
        momentum=0.937,  # 用于 SGD 的动量因子，或用于 Adam 优化器的 beta1，用于将过去的梯度纳入当前更新。
        weight_decay=0.0005,  # L2 正则化项，对大权重进行惩罚，以防止过度拟合。
        warmup_epochs=3.0,  # 学习率预热的历元数，学习率从低值逐渐增加到初始学习率，以在早期稳定训练。
        warmup_momentum=0.8,  # 热身阶段的初始动力，在热身期间逐渐调整到设定动力。
        warmup_bias_lr=0.1,  # 热身阶段的偏置参数学习率，有助于稳定初始历元的模型训练。
        box=7.5,  # 损失函数中边框损失部分的权重，影响对准确预测边框坐标的重视程度。
        cls=0.5,  # 分类损失在总损失函数中的权重，影响正确分类预测相对于其他部分的重要性。
        dfl=1.5,  # 分布焦点损失权重，在某些YOLO 版本中用于精细分类。
        pose=12.0,  # 姿态损失在姿态估计模型中的权重，影响准确预测姿态关键点的重点。
        kobj=2.0,  # 姿态估计模型中关键点对象性损失的权重，在检测可信度和姿态精度之间取得平衡。
        label_smoothing=0.0,  # 应用标签平滑，将硬标签软化为目标标签和标签均匀分布的混合标签，可以提高泛化效果。
        nbs=64,  # 用于损耗正常化的标称批量大小。
        overlap_mask=False,  # 决定在训练过程中分割掩码是否应该重叠，适用于实例分割任务。
        mask_ratio=4,  # 分割掩码的下采样率，影响训练时使用的掩码分辨率。
        dropout=0.0,  # 分类任务中正则化的放弃率，通过在训练过程中随机省略单元来防止过拟合。
        val=True,  # 可在训练过程中进行验证，以便在单独的数据集上对模型性能进行定期评估。
        plots=True  # 生成并保存训练和验证指标图以及预测示例图，以便直观地了解模型性能和学习进度。
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", help="yolov8(n,s,m,l,x)写一个字母就行", default="s", type=str)
    parser.add_argument("-p", "--project_name", help="保存训练结果的项目目录名称", type=str, default="")
    parser.add_argument("-y", "--yaml_path", help="数据集配置文件的路径", type=str, default="")
    parser.add_argument("-e", "--epoch", help="训练轮数", default=200, type=int)
    parser.add_argument("-i", "--img_size", help="输入图片尺寸大小", default=640, type=int)
    parser.add_argument("-cg", "--use_gpu", help="使用GPU", action="store_true", required=False)
    parser.add_argument("-re", "--resume", help="恢复训练", action="store_true", required=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train(get_args())
