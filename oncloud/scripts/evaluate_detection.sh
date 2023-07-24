python evaluate_detection.py --val-skip 5 --dataset coco2017 \
--model resdet50 -b 4 --amp --lr 3e-4 --warmup-lr 5e-4 --warmup-epochs 1 \
--sync-bn --opt adam --root ../datasets/coco --modelpath weights/detection/epoch21_max_0367.pth \
--decay-epochs 5 --decay-rate 0.65
