

python ondevice_searching_sa.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth  --population_size 200 --GPU  