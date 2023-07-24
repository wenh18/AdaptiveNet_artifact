#!/bin/bash

# test_devices=("xiaomi")
test_devices=("jetson_nano" "3090" "xiaomi")
for device in "${test_devices[@]}"
do

    python test_searched_subnets.py --dataset imagenet --model mobilenetv2_100 -b 1024 -j 16 --drop 0.3 \
    --drop-connect 0.2 --num-classes 1000 \
    --data_dir ../datasets/imagenet/val/ \
    --model_path weights/mbv2_100_25_5_1epoch87.pth --GPU \
    --pruned --batch_size_for_lat 32 --test_device $device

    python test_searched_subnets.py --dataset imagenet --model resnet101 -b 512 -j 4 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
    --data_dir ../datasets/imagenet/val/ \
    --model_path weights/resnet101_stage1_epoch58acc75.pth --GPU --use_subset --batch_size_for_lat 32 --test_device $device


    python test_searched_subnets.py --dataset imagenet --model resnet50 -b 512 -j 8 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
    --data_dir ../datasets/imagenet/val/ \
    --model_path weights/resnet1epoch59acc69.pth --GPU --use_subset --batch_size_for_lat 32 --test_device $device

done