# Created Time: 2017年09月07日 星期四 10时07分43秒
#########################################################################
#!/bin/bash

export PATH="/mnt/lustre/pengzhanglin/anaconda2/bin:/mnt/lustre/share/cuda-8.0-cudnn5.1/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-8.0-cudnn5.1/lib64:$LD_LIBRARY_PATH"

#OUTPUT=/mnt/lustre/DATAshare/LSVC2017_features/inceptionv4
OUTPUT=.
LIST=list
SPLIT=train

#for (( batch=238; batch<244; batch++ ))
for batch in 0
do
  srun -p Test \
    -w SZ-IDC1-172-20-20-22 \
    --gres=gpu:1 \
    --job-name=${batch} \
    python -u run.py \
    --output=${OUTPUT} \
    --split=${SPLIT} \
    --label_file=${LIST} \
    --record_id=${batch} \
    2>&1|tee log/$batch.log &
done
