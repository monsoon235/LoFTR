#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh
conda activate loftr

export IN_BITAHUB=1

# 下面改成你的启动命令
bash scripts/reproduce_train/outdoor_ds_840_8_profile.sh > /output/train_840_8_profile.txt 2>&1
