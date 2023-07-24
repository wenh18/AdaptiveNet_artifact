echo "AdaptiveNet-100"
python ondevice_searching_ea.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth \
--searching_times 20 \
--population_size 100 \
--method AdaptiveNet 
wait

echo "AdaptiveNet-200"
python ondevice_searching_ea.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth \
--searching_times 10 \
--population_size 200 \
--method AdaptiveNet 
wait

echo "EA-100"
python ondevice_searching_ea.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth \
--searching_times 20 \
--population_size 100 \
--method BaseLine0 
wait

echo "EA-200"
python ondevice_searching_ea.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth \
--searching_times 10 \
--population_size 200 \
--method BaseLine0 
wait

echo "SA-100"
python ondevice_searching_sa.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth  --population_size 100 --GPU  
wait

echo "SA-200"
python ondevice_searching_sa.py --dataset imagenet --model resnet50 --drop 0.1 --drop-path 0.05 --num-classes 1000 \
--data_dir ./ValForDevice \
--model_path pretrainedweight/resnet1epoch59acc69.pth  --population_size 200 --GPU 
wait