# Created Time: 2017年10月08日
#########################################################################
#!/bin/bash


for ((index_range=41;index_range<=60;index_range++))
do
    python -u run.py \
    --index_range=$index_range \
    2>&1|tee log/$index_range.log
done

