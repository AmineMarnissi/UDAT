#!/bin/bash
source_dataset="voc_0712"
target_dataset="clipart"
net="res101"
s=1
gamma=3.0

python train_net.py --dataset ${source_dataset} --dataset_t ${target_dataset} --net ${net} --s ${s} --epochs 8 --gamma ${gamma}