cd tools
python run_multiprocess.py 32 resnet50 1
wait
python run_multiprocess.py 32 resnet50 2
wait
python run_multiprocess.py 32 resnet50 4
wait
python run_multiprocess.py 64 resnet50 1


python run_multiprocess.py 32 mobilenet_v2 1
wait
python run_multiprocess.py 32 mobilenet_v2 2
wait
python run_multiprocess.py 32 mobilenet_v2 4
wait
python run_multiprocess.py 64 mobilenet_v2 1
