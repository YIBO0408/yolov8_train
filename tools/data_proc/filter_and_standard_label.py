# !/usr/bin/env python

import os
import json
import argparse


def conver_labels_with_dir(json_dir, filter_postfix_list, filter_label_list):
    for dirpath, _, filenames in os.walk(json_dir):
        print("Processing with: ", dirpath)
        for tmp_file in filenames:
            if tmp_file[-4:] == "json":
                with open(os.path.join(dirpath, tmp_file), encoding="utf-8") as tmpf:
                    jfile = json.load(tmpf)
                new_shape = []
                for _, _shape in enumerate(jfile["shapes"]):
                    is_label_remove = False
                    label = _shape.get("label", "")
                    for filter_postfix in filter_postfix_list:
                        if filter_postfix in label:
                            is_label_remove = True
                            break
                    pure_label = label.split("-")[0]
                    if pure_label in filter_label_list:
                        is_label_remove = True

                    if is_label_remove:
                        continue

                    _shape["label"] = pure_label
                    _shape["flags"] ={}
                    new_shape.append(_shape)
                jfile["shapes"] = new_shape

                with open(
                    os.path.join(dirpath, tmp_file), "w", encoding="utf-8"
                ) as new_jf:
                    json.dump(jfile, new_jf, ensure_ascii=False, indent=4)
    print("Label filter and standard is complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop image and json for lupinus dataset"
    )
    parser.add_argument("--json-dir", type=str, default="", help="path for json files")
    parser.add_argument(
        "--filter-postfix-list",
        type=str,
        nargs="+",
        default=["MH-S1", "MH-S2", "KBJ"],
        help="postfix to remove labels",
    )
    parser.add_argument(
        "--filter-label-list",
        type=str,
        nargs="+",
        default=[],
        help="label name to remove",
    )
    args = parser.parse_args()

    conver_labels_with_dir(
        args.json_dir, args.filter_postfix_list, args.filter_label_list
    )
