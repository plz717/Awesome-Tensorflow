# Created Time: 2017年10月08日
#########################################################################
#!/bin/bash

export PATH="/mnt/lustre/panglinzhuo/anaconda2/bin:/mnt/lustre/share/cuda-8.0-cudnn5.1/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-8.0-cudnn5.1/lib64:$LD_LIBRARY_PATH"


train_directory=/mnt/lustre/DATAshare/webvision/resized_images
validation_directory=/mnt/lustre/DATAshare/webvision/resized_images/val_images_256
output_directory=/mnt/lustre/DATAshare/webvision/images_lables_features/tfrecord
labels_file=/mnt/lustre/DATAshare/webvision/info/test_filelist.txt
ckpt_dir=/mnt/lustre/DATAshare/model-zoo/
dataset_name="val"

for ((index_range=0;index_range<=10;index_range++))
do
  srun -p Test \
    -w SZ-IDC1-172-20-20-21 \
    --gres=gpu:1 \
    --job-name="$index_range"-of-"$dataset_name" \
    python -u run_for_validation.py \
    --validation_directory=$validation_directory \
    --output_directory=$output_directory \
    --labels_file=$labels_file \
    --ckpt_dir=$ckpt_dir \
    --dataset_name=$dataset_name \
    --index_range=$index_range \
    2>&1|tee log/$index_range.log &
done

