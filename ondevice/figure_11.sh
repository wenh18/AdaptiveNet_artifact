python ondevice_searching_ea.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth \
--searching_times 1 \
--population_size 50 \
--method BaseLine0 
wait

python ondevice_searching_ea.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth \
--searching_times 1 \
--population_size 100 \
--method BaseLine0 
wait

python ondevice_searching_ea.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth \
--searching_times 1 \
--population_size 200 \
--method BaseLine0 
wait

python ondevice_searching_ea.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth \
--searching_times 1 \
--population_size 50 \
--method AdaptiveNet 
wait

python ondevice_searching_ea.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth \
--searching_times 1 \
--population_size 100 \
--method AdaptiveNet 
wait

python ondevice_searching_ea.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth \
--searching_times 1 \
--population_size 200 \
--method AdaptiveNet 
wait
