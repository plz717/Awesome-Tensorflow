# Created Time: 2017年10月08日
#########################################################################
#!/bin/bash


for ((index_range=21;index_range<=49;index_range++))
do
    python -u run_for_validation.py \
    --index_range=$index_range \
    2>&1|tee log/$index_range.log
done

