#!/bin/bash

# test_devices=("xiaomi")
test_devices=("jetson_nano" "3090" "xiaomi")
for device in "${test_devices[@]}"
do
    python ondevice_searching.py --dataset imagenet --model resnet101 -b 512 -j 4 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
    --data_dir ../datasets/imagenet/val/ \
    --model_path weights/resnet101_stage1_epoch58acc75.pth --GPU --use_subset --batch_size_for_lat 32 --test_device $device

    python ondevice_searching.py --dataset imagenet --model mobilenetv2_100 -b 1024 -j 16 --drop 0.3 \
    --drop-connect 0.2 --num-classes 1000 \
    --data_dir ../datasets/imagenet/val/ \
    --model_path weights/mbv2_100_25_5_1epoch87.pth --GPU \
    --pruned --batch_size_for_lat 32 --test_device $device

    python ondevice_searching.py --dataset imagenet --model resnet50 -b 512 -j 8 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
    --data_dir ../datasets/imagenet/val/ \
    --model_path weights/resnet1epoch59acc69.pth --GPU --use_subset --batch_size_for_lat 32 --test_device $device

done



# python ondevice_searching.py --dataset imagenet --model efficientnetv2_rw_t -b 64 -j 8 --drop 0.3 \
# --drop-connect 0.2 --num-classes 1000 \
# --data_dir ../../project1/MultiBranchNet/mobilenet-yolov4-pytorch/pytorch-image-models-master/data/imagenet/val/ \
# --model_path /home/wenh/Desktop/newproject1/pytorch-image-models-master/output/mbv2_slim/efi120d_small_batch_1epoch41.pth --GPU --pths_path output/resnet_weights/pths_for_device_torchvision/ \
# --use_subset --batch_size_for_lat 32

#python ondevice_searching.py --dataset imagenet --model efficientnetv2_rw_s -b 64 -j 8 --drop 0.3 \
#--drop-connect 0.2 --num-classes 1000 \
#--data_dir ../../project1/MultiBranchNet/mobilenet-yolov4-pytorch/pytorch-image-models-master/data/imagenet/val/ \
#--model_path ~/Desktop/multi_branch_good_results/effi_s_great_lr_1epoch90.pth --GPU --pths_path output/resnet_weights/pths_for_device_torchvision/ \
#--use_subset --batch_size_for_lat 32
#
#python ondevice_searching.py --dataset imagenet --model efficientnetv2_rw_m -b 128 -j 8 --drop 0.3 \
#--drop-connect 0.2 --num-classes 1000 \
#--data_dir ../../project1/MultiBranchNet/mobilenet-yolov4-pytorch/pytorch-image-models-master/data/imagenet/val/ \
#--model_path ~/Desktop/multi_branch_good_results/efi1epoch39.pth --GPU --pths_path output/resnet_weights/pths_for_device_torchvision/ \
#--use_subset --batch_size_for_lat 32
#
#
#python ondevice_searching.py --dataset imagenet --model mobilenetv2_140 -b 256 -j 16 --drop 0.3 \
#--drop-connect 0.2 --num-classes 1000 \
#--data_dir ../../project1/MultiBranchNet/mobilenet-yolov4-pytorch/pytorch-image-models-master/data/imagenet/val/ \
#--model_path output/effi_weights/mbv2_140_exp2/efi120d1epoch138_acc_69_6.pth --GPU --pths_path output/resnet_weights/pths_for_device_torchvision/ \
#--batch_size_for_lat 32 --use_subset
#
#python ondevice_searching.py --dataset imagenet --model mobilenetv2_120d -b 256 -j 16 --drop 0.3 \
#--drop-connect 0.2 --num-classes 1000 \
#--data_dir ../../project1/MultiBranchNet/mobilenet-yolov4-pytorch/pytorch-image-models-master/data/imagenet/val/ \
#--model_path output/effi_weights/mbv2_120d_exp2/efi120d1epoch115_best_70_17.pth --GPU --pths_path output/resnet_weights/pths_for_device_torchvision/ --batch_size_for_lat 32