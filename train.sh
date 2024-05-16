#!/usr/bin/env bash


container_id=($(echo $(docker ps -a | grep yolov8_ocr | awk '{print $1}') | tr " " "\n"))
for i in "${container_id[@]}"
do
    echo "contain_id: "$i
    docker stop $i
    docker rm $i
done

DATA_DIR=""
while [ -z "$DATA_DIR" ]; do
    echo "请输入数据集绝对路径："
    read DATA_DIR
done

if [ -z "$DATA_DIR" ]; then
    echo "错误：未提供数据集路径"
    exit 1
fi


echo "请输入项目名称（默认为ocr，默认直接回车）："
read project_name
project_name=${project_name:-ocr}

echo "请输入模型类型（s或者n,默认为s，默认直接回车）："
read model
model=${model:-s}

echo "请输入检测det还是分割seg（det或者seg,默认为det，默认直接回车）："
read type
type=${type:-det}

if [[ "$type" == "seg" ]]; then
    model_type=$model-seg
else
    model_type=$model
fi

echo "请输入训练轮数（默认为200，默认直接回车）："
read epoch
epoch=${epoch:-200}

echo "是否恢复训练？（是/否，默认为否，默认直接回车）："
read resume_training
resume_training=${resume_training:-否}

if [[ "$resume_training" == "是" ]]; then
    resume_flag="-re"
else
    resume_flag=""
fi

directory_name=$(basename "$DATA_DIR")

run_docker_cpu() {
    docker run \
        --name ocr_train \
        -it \
        --network=host \
        -v $(pwd):$(pwd) \
        -v $DATA_DIR:$DATA_DIR \
        registry.cn-shanghai.aliyuncs.com/epo0408/yolov8_ocr:v1.1 \
        bash -c "
        cd $(pwd) &&
        python3 tools/data_convert/labelme_2_yolov8_det_seg.py --ROOT_DIR $DATA_DIR --modelType $type &&
        chmod 777 -R $DATA_DIR &&
        python3 train/train_det_seg.py -p $project_name -m $model_type  -y $(pwd)/dataset/$directory_name/yolov8_ocr.yaml -e $epoch $resume_flag &&
        chmod 777 -R $project_name dataset"
}

run_docker_gpu() {
    docker run \
        --gpus all \
        --shm-size=4g \
        --name ocr_train \
        -it \
        --network=host \
        -v $(pwd):$(pwd) \
        -v $DATA_DIR:$DATA_DIR \
        registry.cn-shanghai.aliyuncs.com/epo0408/yolov8_ocr:v1.1 \
        bash -c "
        cd $(pwd) &&
        python3 tools/data_convert/labelme_2_yolov8_det_seg.py --ROOT_DIR $DATA_DIR --modelType $type &&
        chmod 777 -R $DATA_DIR &&
        python3 train/train_det_seg.py -p $project_name -m $model_type  -y $(pwd)/dataset/$directory_name/yolov8_ocr.yaml -e $epoch --use_gpu $resume_flag &&
        chmod 777 -R $project_name dataset"
}

echo "请选择运行设备(1为CPU，2为GPU)："
options=("CPU" "GPU")
read -p "请输入选项：" opt_index

if [[ "$opt_index" =~ ^[1-${#options[@]}]$ ]]; then
    selected_option=${options[opt_index-1]}
    if [[ "$selected_option" == "CPU" ]]; then
        run_docker_cpu
    elif [[ "$selected_option" == "GPU" ]]; then
        run_docker_gpu
    else
        echo "无效选项"
    fi
else
    echo "无效选项"
fi
