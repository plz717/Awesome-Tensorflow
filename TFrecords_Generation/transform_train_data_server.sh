# Created Time: 2017年10月08日
#########################################################################
#!/bin/bash

export PATH="/mnt/lustre/panglinzhuo/anaconda2/bin:/mnt/lustre/share/cuda-8.0-cudnn5.1/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-8.0-cudnn5.1/lib64:$LD_LIBRARY_PATH"


train_directory=/mnt/lustre/DATAshare/webvision/resized_images
validation_directory=/mnt/lustre/DATAshare/webvision/resized_images/val_images_256
output_directory=/mnt/lustre/DATAshare/webvision/images_lables_features/tfrecord_with_uid
labels_file=/mnt/lustre/DATAshare/webvision/info/label_file_all.txt
ckpt_dir=/mnt/lustre/DATAshare/model-zoo/


#train_directory=/home/sensetime/plz_workspace/Misc/WebVisionStat/webvision/ResizedImages/
#validation_directory=/home/sensetime/plz_workspace/Misc/WebVisionStat/webvision/ResizedImages/val_images_256/
#output_directory=/home/sensetime/plz_workspace/Misc/WebVisionStat/webvision/zhanglin/tfrecord_split/google/
#labels_file=/home/sensetime/plz_workspace/Misc/WebVisionStat/webvision/info/info/label_file_all.txt
#ckpt_dir=/home/sensetime/plz_workspace/Misc/WebVisionStat/ckpt_files/
dataset_name="flickr"
subset_name="train"



for ((index_range=5;index_range<=7;index_range++))
do
  srun -p Test \
    -w SZ-IDC1-172-20-20-21 \
    --gres=gpu:1 \
    --job-name="$index_range"-of-"$dataset_name" \
    python -u run.py \
    --train_directory=$train_directory \
    --validation_directory=$validation_directory \
    --output_directory=$output_directory \
    --labels_file=$labels_file \
    --ckpt_dir=$ckpt_dir \
    --dataset_name=$dataset_name \
    --index_range=$index_range \
    --subset_name=$subset_name \
    2>&1|tee log/$index_range.log
done

