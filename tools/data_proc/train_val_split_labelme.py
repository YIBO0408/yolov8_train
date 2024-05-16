from pathlib import Path
from typing import Tuple, List, Union
import pandas as pd
import glob
import os
import shutil
import argparse
import json


def json_read(json_path: Union[str, Path]):
    with open(json_path) as f:
        return json.load(f)


def generate_df(json_folder_path: Union[str, Path]):
    """
    :param json_folder_path: 包含json文件的父文件夹
    :return:
    """
    json_folder_path = Path(json_folder_path)
    df = pd.DataFrame(columns=("label", "image"))
    for i in json_folder_path.glob("**/*.json"):
        temp_dict = json_read(i)
        for j in temp_dict["shapes"]:
            df.loc[df.shape[0]] = {
                "label": j["label"],
                "image": i.parent / temp_dict["imagePath"],
            }
    return df


def prepare_sorted_defect_stats(defect_stats, min_val_num, train_percent):
    """原缺陷数据的统计信息要检查修改一下."""
    val_dict = {}
    for label, label_num in defect_stats.items():
        if label_num > min_val_num:
            if label_num * (1 - train_percent) > min_val_num:
                val_dict[label] = int(label_num * (1 - train_percent) + 0.5)
            else:
                val_dict[label] = min_val_num
        else:
            val_dict[label] = label_num
    sorted_defect_stats = sorted(val_dict.items(), key=lambda x: x[1])
    return sorted_defect_stats


def split_val_train(
        sorted_defect_stats: List[Tuple[str, int]],
        df: pd.DataFrame,
        val: Union[str, Path] = "val",
        train: Union[str, Path] = "train",
):
    """只要有了每种类别要多少个验证集数据的信息, 以及全体数据的dataframe, 就可以跑这个程序了."""
    val = Path(val)
    train = Path(train)
    # 保证val的每类缺陷数量不能少到我们指定的数字
    for i in sorted_defect_stats:
        temp_df = df[df["label"] == i[0]]
        for j in temp_df.sort_values(by=["image"])[: i[1]].iterrows():
            try:
                image_path = Path(j[1]["image"])
                # val/文件夹下面的子文件夹, 一律加上`val_`开头
                parent = f"val_{image_path.parent.name}"
                json_path = image_path.with_suffix(".json")
                val_parent = val / parent
                if not val_parent.is_dir():
                    val_parent.mkdir(parents=True)
                shutil.copy(image_path, val_parent / image_path.name)
                shutil.copy(json_path, val_parent / json_path.name)
                # 删除image对应的所有行
                try:
                    del_index_df = df[df["image"] == j[1]["image"]].index
                    df = df.drop(del_index_df)
                except:
                    continue
            except:
                continue
    # 训练集数据copy
    for k in list(set([i[1]["image"] for i in df.iterrows()])):
        image_path = Path(k)
        # train/文件夹下面的子文件夹, 一律加上`train`开头
        parent = f"train_{image_path.parent.name}"
        json_path = image_path.with_suffix(".json")
        train_parent = train / parent
        if not train_parent.is_dir():
            train_parent.mkdir(parents=True)
        shutil.copy(image_path, train_parent / image_path.name)
        shutil.copy(json_path, train_parent / json_path.name)


def train_val_random_split(input_path, output_path, train_percent):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    from sklearn.model_selection import train_test_split
    img_path_list = glob.glob('{}/*.jpg'.format(input_path))
    json_path_list = glob.glob('{}/*.json'.format(input_path))
    train_set, val_set = train_test_split(
        img_path_list, test_size=train_percent, random_state=14
    )
    out_val_dir = '{}/{}/val_{}'.format(output_path, 'val', os.path.basename(input_path))
    out_train_dir = '{}/{}/train_{}'.format(output_path, 'train', os.path.basename(input_path))
    if not os.path.exists(out_val_dir):
        os.makedirs(out_val_dir)
    if not os.path.exists(out_train_dir):
        os.makedirs(out_train_dir)
    for img_path in img_path_list:
        try:
            json_path = img_path.replace('.jpg', '.json')
            if json_path in json_path_list:
                if img_path in val_set:
                    shutil.copy(img_path, '{}/{}'.format(out_val_dir, os.path.basename(img_path)))
                    shutil.copy(json_path, '{}/{}'.format(out_val_dir, os.path.basename(json_path)))
                else:
                    shutil.copy(img_path, '{}/{}'.format(out_train_dir, os.path.basename(img_path)))
                    shutil.copy(json_path, '{}/{}'.format(out_train_dir, os.path.basename(json_path)))
        except:
            continue
    print('划分方式：随机划分，训练集数量：{}张，验证集数量：{}张'.format(len(train_set), len(val_set)))


def train_val_min_split(args):
    labelme_dir = Path(args.input_dir)
    df = generate_df(labelme_dir)
    defect_stats = df["label"].value_counts().to_dict()
    sorted_defect_stats = prepare_sorted_defect_stats(
        defect_stats, args.min_val_num, args.train_percent
    )
    output_path = Path(args.output_dir)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    val_dir = output_path / "val"
    train_dir = output_path / "train"
    split_val_train(sorted_defect_stats, df, val_dir, train_dir)
    print('划分方式：过滤指定量缺陷划分,训练集数量：{}张，验证集数量：{}张'.format(
        len(glob.glob('{}/*/*.json'.format(train_dir))), len(glob.glob('{}/*/*.json'.format(val_dir))))
    )


def main(args) -> None:
    if args.random_flag:
        train_val_random_split(args.input_dir, args.output_dir, args.train_percent)
    else:
        train_val_min_split(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split the detection dataset into train and val sets."
    )
    parser.add_argument("--input-dir", type=str, help="Input dir")
    parser.add_argument("--output-dir", type=str, help="Output dir")
    parser.add_argument(
        "--train-percent", type=float, default=0.8, help="Train percentage for dataset"
    )
    parser.add_argument(
        "--min-val-num",
        type=int,
        default=2,
        help="Min defect number for val dataset. if less than min. all are put in val dataset",
    )
    parser.add_argument(
        "--random-flag", action="store_true", help="random split dataset"
    )
    # Add another argument to specify if delete the original files.
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
