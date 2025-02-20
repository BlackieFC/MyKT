#!/bin/bash
# 设置默认dataset值
default_dataset="arithmetic_retag"

# 检查是否有输入参数
if [ $# -eq 0 ]; then
    # 如果没有输入参数，使用默认值
    dataset=$default_dataset
else
    # 如果有输入参数，使用输入的值
    dataset=$1
fi
echo "参与训练的dataset: $dataset"

# 加载.env文件
env_file="scripts/.env"
if [ -f $env_file ]; then
    export $(cat $env_file | sed 's/#.*//g' | xargs)
fi
echo "TRAIN_PATH: $TRAIN_PATH"
echo "CONDA_PATH: $CONDA_PATH"

source $CONDA_PATH DTransformer
cd $TRAIN_PATH

python train_0102.py \
    -d ${dataset} \
    -bs 32 \
    -tbs 32 \
    --d_model 128 \
    --n_know 32 \
    -o output/CDM_${dataset} \
    --device cuda
