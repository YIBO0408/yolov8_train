#!/usr/bin/env bash

model_path=${1:-/home/yibo/yolov8_train/train_projects/jingshen_ocr/train/weights/best.pt}
export_format=${2:-onnx}

model_dir=$(dirname $model_path)

container_id=($(echo $(docker ps -a | grep yolov8_ocr | awk '{print $1}') | tr " " "\n"))
for i in "${container_id[@]}"

do
echo "contain_id: "$i
docker stop $i
docker rm $i
done

echo 正在转换$model_path

docker run \
--gpus all \
--rm \
--name ocr_export \
-it \
-v $(pwd):$(pwd) \
-v $model_path:$model_path \
registry.cn-shanghai.aliyuncs.com/epo0408/yolov8_ocr:v1.1 \
bash -c "cd $(pwd)/export && python3 export.py -m $model_path -e $export_format && chmod 777 -R $model_dir"