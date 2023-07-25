# AdaptiveNet_artifact

This repo is the PyTorch implementation of the paper AdaptiveNet: Post-deployment Neural Architecture Adaptation for Diverse Edge Environments (MobiCom 2023).

## Environment  
The required packages of the environment we used to conduct experiments are listed in requirements.txt.

## Datatset organization
You can download from [datasets](https://cloud.tsinghua.edu.cn/f/160fb02458304bd39c43/) for test, and organize it as following:
- repo
  - datasets
    - imagenet
    - CamVid
    - coco
  - oncloud
    - mytimm
    - scripts
    - ...
  - ondevice

## Experiments
for finding the optimal subnets:
```shell
cd oncloud
bash scripts/finder.sh
```
for training from scratch:
```shell
cd oncloud
bash scripts/train_cls.sh
```
## Acknowledgements
We would like to thank the code from [timm](https://github.com/huggingface/pytorch-image-models), [EfficientDetV2](https://github.com/rwightman/efficientdet-pytorch), and [segmentation-models.pytorch](https://github.com/qubvel/segmentation_models.pytorch).
