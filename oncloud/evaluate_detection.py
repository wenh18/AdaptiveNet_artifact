#!/usr/bin/env python
""" EfficientDet Training Script

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import time
import yaml
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass
import mytools
from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler
from timm.models import resume_checkpoint, load_checkpoint
from timm.models.layers import set_layer_config
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch.utils.data import Subset
import numpy as np
from PIL import Image
torch.backends.cudnn.benchmark = True


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('--root', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of dataset to train (default: "coco"')
parser.add_argument('--model', default='tf_efficientdet_d1', type=str, metavar='MODEL',
                    help='Name of model to train (default: "tf_efficientdet_d1"')
add_bool_arg(parser, 'redundant-bias', default=None, help='override model config for redundant bias')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--val-skip', type=int, default=0, metavar='N',
                    help='Skip every N validation samples.')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--no-pretrained-backbone', action='store_true', default=False,
                    help='Do not start with pretrained backbone weights, fully random.')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--clip-grad', type=float, default=10.0, metavar='NORM',
                    help='Clip gradient norm (default: 10.0)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Optimizer parameters
parser.add_argument('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "momentum"')
parser.add_argument('--opt-eps', default=1e-3, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=4e-5,
                    help='weight decay (default: 0.00004)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

# loss
parser.add_argument('--smoothing', type=float, default=None, help='override model config label smoothing')
add_bool_arg(parser, 'jit-loss', default=None, help='override model config for torchscript jit loss fn')
add_bool_arg(parser, 'legacy-focal', default=None, help='override model config to use legacy focal loss')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
add_bool_arg(parser, 'bench-labeler', default=False,
             help='label targets in model bench, increases GPU load at expense of loader processes')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='map', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "map"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--modelpath', metavar='DIR', help='path to dataset')
parser.add_argument('--stage', type=int, default=1)

def get_model_size(Detmodel, subnet):
    multimodel = Detmodel.model.backbone
    model = []
    model.append(multimodel.conv1)
    model.append(multimodel.bn1)
    model.append(multimodel.act1)
    model.append(multimodel.maxpool)
    for blockidx in range(len(subnet[0])):
        if subnet[0][blockidx] != 99:
            model.append(multimodel.layer2[blockidx][subnet[0][blockidx]])
    for blockidx in range(len(subnet[1])):
        if subnet[1][blockidx] != 99:
            model.append(multimodel.layer3[blockidx][subnet[1][blockidx]])
    for blockidx in range(len(subnet[2])):
        if subnet[2][blockidx] != 99:
            model.append(multimodel.layer4[blockidx][subnet[2][blockidx]])
    multimodel = Detmodel.model
    model.append(multimodel.fpn)
    model.append(multimodel.class_net)
    model.append(multimodel.box_net)
    temp = torch.nn.Sequential(*model)
    tmp_model_file_path = 'tmp.model'
    torch.save(temp, tmp_model_file_path)
    model_size = os.path.getsize(tmp_model_file_path)
    os.remove(tmp_model_file_path)
    model_size /= 1024 ** 2
    del temp
    return model_size

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_clip_parameters(model, exclude_head=False):
    if exclude_head:
        # FIXME this a bit of a quick and dirty hack to skip classifier head params
        return [p for n, p in model.named_parameters() if 'predict' not in n]
    else:
        return model.parameters()


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.pretrained_backbone = not args.no_pretrained_backbone
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.distributed:
        logging.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        logging.info('Training with a single process on 1 GPU.')

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            logging.warning("Neither APEX or native Torch AMP is available, using float32. "
                            "Install NVIDA apex or upgrade to PyTorch 1.6.")

    if args.native_amp:
        if has_native_amp:
            use_amp = 'native'
        else:
            logging.warning("Native AMP not available, using float32. Upgrade to PyTorch 1.6.")
    elif args.apex_amp:
        if has_apex:
            use_amp = 'apex'
        else:
            logging.warning("APEX AMP not available, using float32. Install NVIDA apex")

    random_seed(args.seed, args.rank)

    with set_layer_config(scriptable=args.torchscript):
        model = create_model(
            args.model,
            bench_task='train',
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            pretrained_backbone=args.pretrained_backbone,
            redundant_bias=args.redundant_bias,
            label_smoothing=args.smoothing,
            legacy_focal=args.legacy_focal,
            jit_loss=args.jit_loss,
            soft_nms=args.soft_nms,
            bench_labeler=args.bench_labeler,
            checkpoint_path=args.initial_checkpoint,
        )
    model_config = model.config  # grab before we obscure with DP/DDP wrappers
    model.model.load_state_dict(torch.load(args.modelpath, map_location=torch.device('cpu')), strict=True)
    # print(model)
    # lats = mytools.get_lats(model)
    if args.local_rank == 0:
        logging.info('Model %s created, param count: %d' % (args.model, sum([m.numel() for m in model.parameters()])))

    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.distributed and args.sync_bn:
        if has_apex and use_amp == 'apex':
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            logging.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model, force native amp with `--native-amp` flag'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model. Use `--dist-bn reduce` instead of `--sync-bn`'
        model = torch.jit.script(model)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            logging.info('Using native Torch AMP. Training in mixed precision.')
    elif use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            logging.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            logging.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            unwrap_bench(model), args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
        if args.resume:
            load_checkpoint(unwrap_bench(model_ema), args.resume, use_ema=True)

    if args.distributed:
        if has_apex and use_amp == 'apex':
            if args.local_rank == 0:
                logging.info("Using apex DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logging.info("Using torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.device])
        # NOTE: EMA model does not need to be wrapped by DDP...
        if model_ema is not None and not args.resume:
            # ...but it is a good idea to sync EMA copy of weights
            # NOTE: ModelEma init could be moved after DDP wrapper if using PyTorch DDP, not Apex.
            model_ema.set(model)

    loader_eval, evaluator = create_datasets_and_loaders(args, model_config)
    # import pdb;pdb.set_trace()

    # if model_config.num_classes < loader_train.dataset.parser.max_label:
    #     logging.error(
    #         f'Model {model_config.num_classes} has fewer classes than dataset {loader_train.dataset.parser.max_label}.')
    #     exit(1)
    # if model_config.num_classes > loader_train.dataset.parser.max_label:
    #     logging.warning(
    #         f'Model {model_config.num_classes} has more classes than dataset {loader_train.dataset.parser.max_label}.')
    latencys = [0.0013505160206496113, [[0.002749936749236752, 0.0027569592601121074, 0.0027531325215041035], [0.0024147708006579467, 0.0024153126610649955, 0.0020566660948474], [0.0024782840651695177, 0.0020377660038495304, 0.002047897589327109], [0.002727144896382033, 0.0027433910755196, 0.0027275567102913903], [0.0014948363255972814, 0.0014833994586058337, 0.0014968544545799795], [0.0014858390345717921, 0.0014849455669672803], [0.0014843988900232797]], [[0.0019092318987605549, 0.0019139496967045947, 0.0019048416253292198], [0.001112234712851168, 0.001113807312165848, 0.0011128006559429746], [0.00111214801518604, 0.0011126152192703401, 0.0011093207079954821], [0.0011120492761785333, 0.0011246782360654888, 0.0011179784331658874], [0.0011128126972853535, 0.0011120805836687185], [0.001109693989609227]], [[0.001725912094116211, 0.0017217361565792198, 0.0017308177370013614], [0.001049742554173325, 0.0010378144004128196], [0.0010321405198838976]], 0.011298762427435981]
    eval_metric = args.eval_metric
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    subnets = mytools.get_val_subnets(model.model)
    with open('visualization/detection/subnets.yaml', 'w') as f:
        yaml.dump(subnets, f)
    lats, maps, sizes = [], [], []
    for subnet in subnets:
        map, lat = validate(model, loader_eval, args, subnet, evaluator, lats=latencys)
        lats.append(lat)
        maps.append(map)
        print('\n\n\n==\n', f'mAP: {map}, latency: {lat}, subnet_architecture: {subnet}', '\n==\n\n\n')
    print(lats, ",", maps)
    with open('visualization/detection/maps.yaml', 'w') as f:
        yaml.dump(maps, f)
    # print(sizes, ",", maps)
    # subnet = [[2, 99, 99, 2, 99, 99, 0], [2, 99, 99, 2, 99, 99], [2, 99, 99]]
    # eval_metrics, latency = validate(model, loader_eval, args, subnet, evaluator)
    # subnet = model.model.generate_main_subnet()
    # eval_metrics, latency = validate(model, loader_eval, args, subnet, evaluator)


def create_datasets_and_loaders(
        args,
        model_config,
        transform_train_fn=None,
        transform_eval_fn=None,
        collate_fn=None,
        # sample_num=None
):
    """ Setup datasets, transforms, loaders, evaluator.

    Args:
        args: Command line args / config for training
        model_config: Model specific configuration dict / struct
        transform_train_fn: Override default image + annotation transforms (see note in loaders.py)
        transform_eval_fn: Override default image + annotation transforms (see note in loaders.py)
        collate_fn: Override default fast collate function

    Returns:
        Train loader, validation loader, evaluator
    """
    input_config = resolve_input_config(args, model_config=model_config)

    dataset_train, dataset_eval = create_dataset(args.dataset, args.root)

    # if sample_num is not None:
    #     idxs = np.random.choice(5000, sample_num, replace=False).tolist()
    #     dataset_eval = Subset(dataset_eval, idxs)
    # setup labeler in loader/collate_fn if not enabled in the model bench
    labeler = None
    if not args.bench_labeler:
        labeler = AnchorLabeler(
            Anchors.from_config(model_config), model_config.num_classes, match_threshold=0.5)

    if args.val_skip > 1:
        dataset_eval = SkipSubset(dataset_eval, args.val_skip)
    loader_eval = create_loader(
        dataset_eval,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_eval_fn,
        collate_fn=collate_fn,
    )

    evaluator = create_evaluator(args.dataset, loader_eval.dataset, distributed=args.distributed, pred_yxyx=False)

    return loader_eval, evaluator


def validate(model, loader, args, subnet, evaluator=None, log_suffix='', lats=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    latency = lats[0] + lats[-1]
    for layeridx in range(len(subnet)):
        for blockidx in range(len(subnet[layeridx])):
            if subnet[layeridx][blockidx] != 99:
                latency += lats[layeridx+1][blockidx][subnet[layeridx][blockidx]]
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            # subnet = model.model.generate_random_subnet()
            # if batch_idx >= 10:
            #     t1 = time.time()
            output = model(input, target, subnet=subnet, args=args)
            # output['detections']:[batchsize, 100, 6] 6->[x_min, y_min, x_max, y_max, score, class]
            loss = output['loss']
            # import pdb;pdb.set_trace()
            if evaluator is not None:
                evaluator.add_predictions(output['detections'], target)
            # import pdb;pdb.set_trace()
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m, loss=losses_m))
    # import pdb;pdb.set_trace()
    metrics = OrderedDict([('loss', losses_m.avg)])
    if evaluator is not None:
        metrics['map'] = evaluator.evaluate()

    print(metrics['map'])
    return metrics['map'], latency


if __name__ == '__main__':
    main()

# [0.03876045496776851, 0.03319692130040641, 0.03387059587420839, 0.037716386294124105, 0.03876045496776851, 0.03410808726994678, 0.03765002886454265, 0.03876045496776851, 0.03876045496776851, 0.03876045496776851, 0.03876045496776851, 0.03876045496776851, 0.032622794912318995, 0.03653168196629997, 0.03876045496776851, 0.03876045496776851, 0.03876045496776851, 0.03876045496776851, 0.03876045496776851, 0.03410808726994678, 0.03765293805286138, 0.03318504372028389, 0.03578019142150879, 0.03876045496776851, 0.03876045496776851, 0.03765002886454265, 0.037275162610140716, 0.0372818648213088, 0.03876045496776851, 0.03876045496776851, 0.03764987955189715, 0.03876045496776851, 0.03653168196629997, 0.03876045496776851, 0.035796572463681, 0.03876045496776851, 0.03653276568711406, 0.03410808726994678, 0.037660271230370114, 0.03189092212253147, 0.035298788186275604, 0.036606352738659795, 0.03876045496776851, 0.03617497887274232, 0.03431510443639274, 0.037275162610140716, 0.03876045496776851, 0.03876045496776851, 0.03764987955189715, 0.03876045496776851, 0.03876045496776851, 0.03876045496776851, 0.03876045496776851, 0.03876045496776851, 0.03765002886454265, 0.03770653647605819, 0.03559279201006649, 0.03876045496776851, 0.03726317906620527, 0.03726317906620527, 0.03876045496776851, 0.035492754945851336, 0.032031109838774705, 0.03653276568711406, 0.03876045496776851, 0.03188039798929234, 0.03147444098886817, 0.03431510443639274, 0.030156530515112055, 0.03653168196629997, 0.03653168196629997, 0.03188039798929234, 0.03578019142150879, 0.035433302021989924, 0.03876045496776851, 0.03765293805286138, 0.03276016977098253, 0.036543877437861284, 0.0333724431317262, 0.031891509740039556, 0.03542639751626988, 0.03410808726994678, 0.036543877437861284, 0.03876045496776851, 0.03549980876421688, 0.0372818648213088, 0.02866264304729423, 0.02734826791166055, 0.03628271276300604, 0.036605810878252744, 0.031090861619120894, 0.03262457221445412, 0.03616473650691485, 0.03004180301319469, 0.036596110372832336, 0.03170033898016419, 0.033987252399174855, 0.03876045496776851, 0.037660271230370114, 0.03411836816806986, 0.037716386294124105, 0.03660480422202987, 0.035238644089361636, 0.03239200572774868, 0.03578019142150879, 0.036171289405437436, 0.03131940148093484, 0.034785436861442795, 0.036352706677985916, 0.03770653647605819, 0.037275162610140716, 0.03876045496776851, 0.03617028274921456, 0.03559279201006649, 0.034785436861442795, 0.033760053942901914, 0.036164587194269356, 0.03207446830441253, 0.033441206421514956, 0.03765293805286138, 0.036352706677985916, 0.033271035762748334, 0.03505858508023349, 0.028322186132874152, 0.03765002886454265, 0.02875157076903064, 0.03096814829893786, 0.03149162639271129, 0.028091748555501304, 0.030902376078596017, 0.03668347753659643, 0.035583293799198036, 0.02992516334610756, 0.03208649519718055, 0.03876045496776851, 0.029112673769093524, 0.030536538422709765, 0.0333724431317262, 0.03319692130040641, 0.03208649519718055, 0.037275162610140716, 0.03130750945120146, 0.03653168196629997, 0.03400232093502776, 0.03131940148093484, 0.03412501739733147, 0.031414109047013104, 0.028931617736816403, 0.03437718718942969, 0.0323853035165806, 0.03171162894277862, 0.03503548978555082, 0.03179361844303632, 0.03524112460589168, 0.03373151836973248, 0.03393995641457914, 0.03559279201006649, 0.028065238336119985, 0.03517519584809891, 0.036543877437861284, 0.03449260827266809, 0.0306577104510683, 0.03245609697669444, 0.03517519584809891, 0.03159148765332771, 0.032712770230842354, 0.03413476365985292, 0.036164587194269356, 0.03208940438549928, 0.03258084287547102, 0.031524388477055715, 0.037275162610140716, 0.03513027923275726, 0.035238644089361636, 0.03375583224826389, 0.03617497887274232, 0.0316997513626561, 0.0351711306909118, 0.034481209937972254, 0.03411821885542436, 0.035172137347134674, 0.03208649519718055, 0.028005070156521268, 0.030394021910850444, 0.031884455921674014, 0.026674593337858567, 0.030406217382411765, 0.029662736738571012, 0.029857722195712002, 0.026512206202805644, 0.03370321399033671, 0.027568318627097393, 0.030604519025243894, 0.0270411486577506, 0.0335581832461887, 0.032519545217957156, 0.030835741698139848, 0.03240538847566855, 0.027608724555583913, 0.03356454589150169, 0.025928092725349197, 0.027327313567652844, 0.03171162894277862, 0.02973447423992735, 0.033947807369810165, 0.028464661704169378, 0.030682036370942085, 0.036219110392560865, 0.035238644089361636, 0.03876045496776851, 0.031641822872739846, 0.030772881074385208, 0.03148527097220372, 0.03524112460589168, 0.034000257048944035, 0.03195958908158119, 0.03319692130040641, 0.03400884734259711, 0.03077797456221147, 0.03468614636045514, 0.036605810878252744, 0.03405908141473327, 0.03401679462856716, 0.03107881786847355, 0.03615566215129814, 0.03410808726994678, 0.03239200572774868, 0.02844787607289324, 0.030479686428802177, 0.035172137347134674, 0.027404262561990755, 0.03189979900013317, 0.03257832141837688, 0.0315907531314426, 0.030833896964487403, 0.03480412261654632, 0.03265253702799479, 0.03308662982902141, 0.0361527529629794, 0.03395233973108157, 0.035182529025607645, 0.03251249139959162, 0.031414109047013104, 0.0274069622309521, 0.02610046694977115, 0.028079553083939984, 0.034459846188323665, 0.02660176729915118, 0.028002035738241792, 0.03208649519718055, 0.035492754945851336, 0.02837242983808421, 0.02977990622472281, 0.03038039592781452, 0.03276016977098253, 0.0254009131229285, 0.03290213719762936, 0.031144753851071752, 0.029790148590550278, 0.029564845441567775, 0.034455788255941984, 0.026589723548503835, 0.028762460959078088, 0.02438004811604818, 0.028063200940989484, 0.031229992105503274, 0.03151236880909313, 0.026961001482876865, 0.02756623788313432, 0.033753351731733834, 0.029478643879745945, 0.027551133223254268, 0.03258084287547102, 0.027326745216292563, 0.03616473650691485, 0.031508835879239167, 0.03251496469131624, 0.02912883806710291, 0.031153329695113977, 0.031969831447408656, 0.02899884936785457, 0.03140659765763716, 0.03363107189987645, 0.029061209071766247, 0.0305995676252577, 0.030054152613938453, 0.03271962175465593, 0.032408297663987284, 0.030576185746626416, 0.029929221278489238, 0.033165303143587975, 0.03157950410939226, 0.030844331991792928, 0.028811449956412265, 0.029923032028506498, 0.03265002038743761, 0.03290827105743716, 0.03409551610850325, 0.02842309017374058, 0.029855518630056668, 0.03152819594951591, 0.0274069622309521, 0.02992516334610756, 0.028678744730323252, 0.02817744438094322, 0.028318128200492475, 0.02592403479296752, 0.026969126980714124, 0.026507064549609865, 0.028892254588579892, 0.03164696452593563, 0.031225471785574246, 0.02698787535079802, 0.028941467554882316, 0.03250843346720994, 0.028703956892996124, 0.02610046694977115, 0.02906777641989968, 0.03147444098886817, 0.03003474919482915, 0.031699645398843165, 0.02764063652115639, 0.026986071557709664, 0.02762871318393283, 0.02586297555403276, 0.026953656264025752, 0.03543149822890157, 0.026523101209390044, 0.02843899919529154, 0.02885709868537055, 0.030598837919909545, 0.027898316431527185, 0.02770277948090524, 0.02885709868537055, 0.035055478413899735, 0.02729979187551171, 0.02582592193526451, 0.02930800120035807, 0.025839576817522147, 0.030487267657963914, 0.028421690969756153, 0.03153944497156625, 0.026953656264025752, 0.030592128483936037, 0.028426148674704808, 0.030392319265038072, 0.02910282395102761, 0.03170033898016419, 0.029340103419140128, 0.036542511949635516, 0.03160060294950851, 0.03271268112490876, 0.031422978699809374, 0.02841184115169024, 0.0305995676252577, 0.030115806695186726, 0.03263094208457253, 0.029913182210440585, 0.032719472442010435, 0.029523423223784474, 0.03509767850240071, 0.024368004365400833, 0.028459560991537694, 0.026585665616122158, 0.029113545562281753, 0.030290772216488616, 0.025783350973418266, 0.027617490652835724, 0.026512206202805644, 0.026584581895308056, 0.025783350973418266, 0.030491669972737626, 0.031392215478299845, 0.02678012125419848, 0.027617490652835724, 0.024727404719651347, 0.031847002530338786, 0.0290195725180886, 0.023706373542246195, 0.026595411878643612, 0.028431945376925997, 0.027640364386818626, 0.027556274876450047, 0.028434575206101542, 0.03016837437947591, 0.024812693547720862, 0.027311775419447158, 0.029304853593460235, 0.026969189595694495, 0.02859624708541716, 0.031773759861185094, 0.02973007433342211, 0.02790779537624783, 0.027304892588143396, 0.02480578904200082, 0.028984467188517254, 0.03189473681979709, 0.027982890003859395, 0.02741555252460518, 0.03050249995607318, 0.030151311797325057, 0.030026522549715908, 0.031874514589406024, 0.026965054598721592, 0.03152226920079703, 0.030208826065063473, 0.03153110513783464, 0.030963035544963796, 0.028365693911157473, 0.02901523041002678, 0.0304094444621693, 0.02842309017374058, 0.030124146528918334, 0.02948899220938634, 0.031609438886546125, 0.028945236495046902, 0.03262400145482535, 0.03160109905281452, 0.03400001140555949, 0.03012565651325264, 0.03154250347253048, 0.024368004365400833, 0.02734826791166055, 0.023706373542246195, 0.027206618376452513, 0.028079553083939984, 0.0254009131229285, 0.026584581895308056, 0.02610046694977115, 0.024368004365400833, 0.026507064549609865, 0.024368004365400833, 0.026596777366869377, 0.026974268633909903, 0.03036401488564231, 0.02482457112784338, 0.025484398157909663, 0.02549644190855701, 0.027908558797354648, 0.026879840426974826, 0.026605654244471075, 0.02585510774092241, 0.025837549055465543, 0.025474155792082197, 0.028685711851023664, 0.024809112452497388, 0.03012565651325264, 0.02438004811604818, 0.027813030011726147, 0.026951151664810952, 0.024809112452497388, 0.02763467605667885, 0.027326725950144754, 0.025845257922856496, 0.026523269788183347, 0.030112897506867995, 0.03264118445039999, 0.028348568714026245, 0.026881617729109946, 0.029707540165294304, 0.032882107628716364, 0.029101458462801846, 0.028500214971677217, 0.029526332412103205, 0.02840507632554179, 0.027327313567652844, 0.02731689539822665, 0.027304500040381843, 0.02729465022231593, 0.027133375707298818, 0.027314174054849025, 0.028435073717676024, 0.029989305168691305, 0.028426148674704808, 0.030113904163090868, 0.02841184115169024, 0.028358242728493428, 0.030765991018276023, 0.031535801261362406, 0.028801600138346352, 0.028358242728493428, 0.025474155792082197, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.02438004811604818, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.025845433726455227, 0.024368004365400833, 0.025479297445277976, 0.024798870086669922, 0.025490192451862376, 0.026511057458742698, 0.0275551261323871, 0.0254009131229285, 0.02438004811604818, 0.025474155792082197, 0.0254009131229285, 0.025403107055509935, 0.02688608988366946, 0.02548382980654938, 0.02583703368601173, 0.025845257922856496, 0.02548382980654938, 0.024810889754632505, 0.02798775229791198, 0.028371326851122307, 0.02887347972754276, 0.029116079060718267, 0.02619834861370048, 0.024715777599450314, 0.02583692049739337, 0.025821929026131677, 0.029448928255023377, 0.027629163530137803, 0.028512198515612666, 0.0248040117398657, 0.025189300980230776, 0.026506888746011134, 0.028435073717676024, 0.028444923535741937, 0.026595411878643612, 0.028931049385456122, 0.02787317410864011, 0.030115806695186726, 0.02731630778071856, 0.028358242728493428, 0.027651885543206727, 0.028816880601825134, 0.03042067903460878, 0.028426148674704808, 0.027304324236783112, 0.02893538667698099, 0.027304324236783112, 0.027304324236783112, 0.02841475034000897, 0.02841475034000897, 0.027304324236783112, 0.027304324236783112, 0.026444981796572904, 0.024368004365400833, 0.024368004365400833, 0.0254009131229285, 0.024368004365400833, 0.0254009131229285, 0.024368004365400833, 0.0254009131229285, 0.024368004365400833, 0.0254009131229285, 0.024715777599450314, 0.024368004365400833, 0.02547814870121503, 0.024368004365400833, 0.0248040117398657, 0.024368004365400833, 0.025474155792082197, 0.025412956873575848, 0.02547814870121503, 0.027895946695347024, 0.028418037626478408, 0.03002943173803464, 0.026506888746011134, 0.02369271865998856, 0.025494529743387243, 0.02480578904200082, 0.024384385407573046, 0.02899316826252022, 0.02547814870121503, 0.024715777599450314, 0.024717554901585434, 0.02694202914382472, 0.024810321403272224, 0.027772860093550247, 0.02891332452947443, 0.02539106330486259, 0.024384385407573046, 0.026950368977556326, 0.02539106330486259, 0.026517588682848998, 0.027304324236783112, 0.03004581278020685, 0.026523269788183347, 0.029007752736409504, 0.02583160304059886, 0.02583881098814685, 0.02892970557164664, 0.027316201816905628, 0.027304324236783112, 0.025832171391959143, 0.028426148674704808, 0.028801600138346352, 0.030490672949588658, 0.027304324236783112, 0.027304324236783112, 0.02843599849277072, 0.028426148674704808, 0.028426148674704808, 0.030054152613938453, 0.027304324236783112, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.024368004365400833, 0.024384385407573046, 0.02547814870121503, 0.024368004365400833, 0.025484398157909663, 0.024715777599450314, 0.025484398157909663, 0.024717554901585434, 0.024368004365400833, 0.024368004365400833, 0.025500210848721592, 0.026501207640676785, 0.024809112452497388, 0.025494529743387243, 0.02547814870121503, 0.025500779200081876, 0.02369271865998856, 0.02583338034273398, 0.02369271865998856, 0.024809112452497388, 0.025500210848721592, 0.026281265297321357, 0.027304892588143396, 0.024715777599450314, 0.026281265297321357, 0.025484398157909663, 0.025832171391959143, 0.024384385407573046, 0.0254074443470348, 0.026507457097371418, 0.02583160304059886, 0.026523269788183347, 0.027304324236783112, 0.026275584191987007, 0.027298643131448763, 0.02583160304059886, 0.026506888746011134, 0.027304324236783112, 0.027304892588143396, 0.02583160304059886, 0.027304324236783112, 0.027304324236783112, 0.027304324236783112, 0.027304324236783112, 0.027304324236783112, 0.027304324236783112, 0.027304324236783112, 0.027304324236783112, 0.027304324236783112, 0.027304324236783112, 0.024368004365400833] , [0.415115633348227, 0.40090290289293945, 0.40505927977227024, 0.4104887351516821, 0.415115633348227, 0.39278211630790516, 0.4101232106248925, 0.415115633348227, 0.415115633348227, 0.415115633348227, 0.415115633348227, 0.415115633348227, 0.3905238320330087, 0.4015052666794827, 0.415115633348227, 0.415115633348227, 0.415115633348227, 0.415115633348227, 0.415115633348227, 0.39278211630790516, 0.40943472633200423, 0.40339320170810783, 0.39570163646687473, 0.415115633348227, 0.415115633348227, 0.4101232106248925, 0.4123915879920905, 0.4077530907344558, 0.415115633348227, 0.415115633348227, 0.41154237816152206, 0.415115633348227, 0.4015052666794827, 0.415115633348227, 0.40407936349661705, 0.415115633348227, 0.4035582557508864, 0.39278211630790516, 0.40838636794123573, 0.38348521475255987, 0.407172358274731, 0.40157830489250923, 0.415115633348227, 0.4031214107114068, 0.384894598913546, 0.4123915879920905, 0.415115633348227, 0.415115633348227, 0.41154237816152206, 0.415115633348227, 0.415115633348227, 0.415115633348227, 0.415115633348227, 0.415115633348227, 0.4101232106248925, 0.40939111389228505, 0.40534399189995246, 0.415115633348227, 0.4103967193779752, 0.4103967193779752, 0.415115633348227, 0.3983363061843004, 0.37815226296939286, 0.4035582557508864, 0.415115633348227, 0.38148791869715326, 0.36004168724807994, 0.384894598913546, 0.3873926175697664, 0.4015052666794827, 0.4015052666794827, 0.38148791869715326, 0.39570163646687473, 0.39690280240535886, 0.415115633348227, 0.40943472633200423, 0.40175354105588007, 0.40006470625853074, 0.39149512069131903, 0.3792797019790075, 0.39982087181651094, 0.39278211630790516, 0.40006470625853074, 0.415115633348227, 0.39461816421123047, 0.4077530907344558, 0.37191326752467463, 0.35699909379913397, 0.41345395101350996, 0.40486354660738944, 0.3862439441190216, 0.3903200550417065, 0.4063618856928509, 0.3717423407207649, 0.4029088196762998, 0.3882381020489973, 0.38943176764465615, 0.415115633348227, 0.40838636794123573, 0.3994204063571211, 0.4104887351516821, 0.4053506068280566, 0.4059053891223319, 0.3991890281335422, 0.39570163646687473, 0.40237176674651415, 0.3931995891758005, 0.40514158796952926, 0.4130922255535076, 0.40939111389228505, 0.4123915879920905, 0.415115633348227, 0.4031128122936309, 0.40534399189995246, 0.40514158796952926, 0.3984815422768958, 0.4052571166438414, 0.3967017099960607, 0.3960295143916405, 0.40943472633200423, 0.4130922255535076, 0.3809675668504687, 0.39689905660581654, 0.3613284168884532, 0.4101232106248925, 0.37076929403341247, 0.38653064095732453, 0.3756729189526741, 0.36900320137741377, 0.39426125856671806, 0.3998872983367551, 0.3897569666533099, 0.3823862218189, 0.3939973152277485, 0.415115633348227, 0.38036097973278765, 0.3889008184325898, 0.39149512069131903, 0.40090290289293945, 0.3939973152277485, 0.4123915879920905, 0.3864102502242618, 0.4015052666794827, 0.3917962263616522, 0.3931995891758005, 0.39991138621343414, 0.3787275231043607, 0.37085942114426373, 0.3899005790359007, 0.4019536296582311, 0.3947606501532432, 0.39445270040092884, 0.38979894192322495, 0.40734172268465674, 0.39962219522406117, 0.3904125646582309, 0.40534399189995246, 0.3772830077201903, 0.4037633041462208, 0.40006470625853074, 0.3972785396325764, 0.3879700567533227, 0.3756411498930807, 0.4037633041462208, 0.38994361313416476, 0.39643620251877715, 0.3974029025959165, 0.4052571166438414, 0.3936456768217103, 0.39167890310601494, 0.38338644628507096, 0.4123915879920905, 0.3976598202628947, 0.4059053891223319, 0.4047729674822191, 0.4031214107114068, 0.39652246475178626, 0.40884684499320345, 0.4009586855474933, 0.3979273939321303, 0.4056329216580165, 0.3939973152277485, 0.370652185667732, 0.3735767408314496, 0.38092869488190456, 0.3494871597812677, 0.37451803387271765, 0.36416833846761826, 0.3780909781907196, 0.36278880243604505, 0.381883808099279, 0.3744818635156224, 0.38925689410685055, 0.3673499547561641, 0.3824915875800792, 0.3741555540159616, 0.38077905185644784, 0.3833637996183575, 0.3703292719014546, 0.390602710646574, 0.35638497538319913, 0.36242942764260144, 0.3947606501532432, 0.37291080889066663, 0.39051884368601864, 0.3690199260467118, 0.38414994133624053, 0.40326196916393753, 0.4059053891223319, 0.415115633348227, 0.3910079726464973, 0.37370747215420264, 0.3654346287232998, 0.40734172268465674, 0.3889580824841425, 0.3837093927705268, 0.40090290289293945, 0.3915226114083888, 0.3825446602292968, 0.3984357189462795, 0.40486354660738944, 0.40162278994588274, 0.39198599969417847, 0.3771476249489849, 0.4013770152971151, 0.39278211630790516, 0.3991890281335422, 0.3742646302451766, 0.3728313775280361, 0.4056329216580165, 0.3613952463470437, 0.3868843571024309, 0.3944321566055029, 0.3837655131959429, 0.38051784848901693, 0.402593673143756, 0.3928930921460144, 0.39428296852929984, 0.4030644611786343, 0.3895858493156857, 0.4036138948747523, 0.37888052086516055, 0.3787275231043607, 0.3668175508338208, 0.3371684048823523, 0.37143147443241425, 0.3873706160913507, 0.36407558435968507, 0.3764616203115779, 0.3939973152277485, 0.3983363061843004, 0.3734567283640333, 0.3799670778199955, 0.3621130463972851, 0.40175354105588007, 0.34452368830005536, 0.3840822385095365, 0.37935811704771466, 0.37816031200917816, 0.37499666907505275, 0.38655690078790816, 0.3518023771781681, 0.3794573486721966, 0.3426159540687372, 0.377523293367217, 0.38956869375082814, 0.38399019423282493, 0.3655609707114169, 0.3649690185307083, 0.4013488883244955, 0.3826555061727749, 0.3641660962087955, 0.39167890310601494, 0.3651449318279943, 0.4063618856928509, 0.3753074536664592, 0.38665962886678945, 0.3838083684172928, 0.39089432524299433, 0.3847552492834682, 0.3693901585551621, 0.3712343992055421, 0.3944911580023956, 0.37939361198814386, 0.3894593396325503, 0.38050035636595375, 0.3931277770791674, 0.3835510489207532, 0.3893260321959013, 0.3826370774458701, 0.3885427101127407, 0.3864697832514281, 0.38532020116747256, 0.37774403824970487, 0.3834220951562644, 0.3909644043946542, 0.38221378569996783, 0.3980189715936369, 0.37727987006246144, 0.3828637894406186, 0.3858983597647778, 0.3668175508338208, 0.3823862218189, 0.3801263220802809, 0.358879342003728, 0.362342142614412, 0.3572328713807504, 0.3604522421383658, 0.35919005084909084, 0.3718124036530287, 0.3951594435025455, 0.37667777638117816, 0.3623108196465814, 0.3711154532569227, 0.37825078630337333, 0.35873502450219497, 0.3371684048823523, 0.3638434259130033, 0.36004168724807994, 0.3740384927867691, 0.3929850201417393, 0.3751454319351072, 0.3590616239735758, 0.3683009996501575, 0.35135750700151025, 0.37032989564391644, 0.3928980407092091, 0.3685437272247093, 0.3738309068504199, 0.35777533114832194, 0.38858439661289385, 0.36066130209734715, 0.36417102819271946, 0.35777533114832194, 0.3944698276665497, 0.372143677878148, 0.3630522962337118, 0.37096795854008136, 0.35915607884798567, 0.37778978215299225, 0.3756323158832376, 0.38578565117214075, 0.37032989564391644, 0.38714292281595764, 0.3767813160347283, 0.37716248070470565, 0.3801665227769518, 0.3882381020489973, 0.3719593373269807, 0.40335475733195625, 0.38668253921307016, 0.3939644113151225, 0.3771885151628172, 0.37991448090614477, 0.3894593396325503, 0.38323802417933867, 0.39300123629902506, 0.383972214882431, 0.39330333830453845, 0.38598636257079283, 0.398862004777516, 0.3296058890286615, 0.3680831111496901, 0.35519627870452464, 0.3820123443686634, 0.35651653060146904, 0.3525559650976147, 0.3624818545348257, 0.36278880243604505, 0.3484927544945285, 0.3525559650976147, 0.3763665744896034, 0.36972752830535577, 0.33699116715576605, 0.3624818545348257, 0.35051381282662913, 0.36862830636349886, 0.3684639141965703, 0.33357734281198764, 0.35725800036533706, 0.3744594961862548, 0.3747714695710879, 0.36550000709726577, 0.374237922919455, 0.38772168996625805, 0.350951694225569, 0.3722248316957469, 0.36881375249588877, 0.36170197150428046, 0.36387921113980104, 0.3728242022108565, 0.36896854047571603, 0.36140234152781187, 0.3689340498266667, 0.3516948918297654, 0.36880675750445113, 0.3878788319377006, 0.3610949792797622, 0.36701169023957014, 0.3760568850103973, 0.3869114262637945, 0.3741257427040964, 0.38748520169939993, 0.3704165358019688, 0.3877837182298848, 0.3884246109273972, 0.3885304168555539, 0.3904947392690161, 0.37651688845998355, 0.37423275055611543, 0.38192232935820863, 0.37727987006246144, 0.3811035441893889, 0.3824039371263417, 0.3854234072875972, 0.37215668026362797, 0.391249956177526, 0.3879252526600188, 0.39371099602644816, 0.38226112132245016, 0.3830684284478958, 0.3296058890286615, 0.35699909379913397, 0.33357734281198764, 0.34977257043362137, 0.37143147443241425, 0.34452368830005536, 0.3484927544945285, 0.3371684048823523, 0.3296058890286615, 0.35919005084909084, 0.3296058890286615, 0.35263125507573856, 0.3629570588284643, 0.35463815462687026, 0.3482603284112741, 0.34332298067657535, 0.35530044817112094, 0.36062034426922207, 0.3663384481158812, 0.3570153216376793, 0.3587602583618885, 0.3640568096562233, 0.3443874624841344, 0.3848175205234366, 0.34576883756964155, 0.38226112132245016, 0.3426159540687372, 0.35090234908583384, 0.3680724211352895, 0.34576883756964155, 0.37714679264870293, 0.3687020146458159, 0.3624486108456257, 0.3703849625237802, 0.38241804200452617, 0.3910442881488338, 0.3749130733314442, 0.3699995883690524, 0.3760456882723522, 0.38582443513761033, 0.3794827888308496, 0.3753592678794073, 0.38525651846362446, 0.3764967575605404, 0.36242942764260144, 0.36578311505310235, 0.36802790655865375, 0.3700228650953041, 0.35537783046547916, 0.3677524157572839, 0.37784114516211936, 0.3805829503442058, 0.3767813160347283, 0.3841581664845073, 0.37991448090614477, 0.3757229257031142, 0.37928167641084276, 0.387316288910528, 0.37641841311980484, 0.3757229257031142, 0.3443874624841344, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3426159540687372, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3590042774116926, 0.3296058890286615, 0.3473969525477708, 0.3455370898842145, 0.3580477173609531, 0.36185558364055787, 0.3659907241116849, 0.34452368830005536, 0.3426159540687372, 0.3443874624841344, 0.34452368830005536, 0.3587699156984889, 0.36032849997384425, 0.3492968281208629, 0.36121505965699524, 0.3624486108456257, 0.3492968281208629, 0.3491531195964877, 0.3589203591107925, 0.3743296645608107, 0.36904887485396554, 0.3821914665382269, 0.3586541343423555, 0.3513729837370441, 0.36101846370766, 0.3610905549282021, 0.36725413483441227, 0.378461303128585, 0.37484576642922635, 0.34834740506995737, 0.34434710215543385, 0.36349315411130856, 0.37784114516211936, 0.37814445349053705, 0.35725800036533706, 0.37411562956424677, 0.3540487983671986, 0.38323802417933867, 0.37126204191303297, 0.3757229257031142, 0.3772702665280603, 0.37770234999180147, 0.3784891283706468, 0.3767813160347283, 0.3722058175713312, 0.3744577347023926, 0.3722058175713312, 0.3722058175713312, 0.3796194849094212, 0.3796194849094212, 0.3722058175713312, 0.3722058175713312, 0.34942724752970966, 0.3296058890286615, 0.3296058890286615, 0.34452368830005536, 0.3296058890286615, 0.34452368830005536, 0.3296058890286615, 0.34452368830005536, 0.3296058890286615, 0.34452368830005536, 0.3513729837370441, 0.3296058890286615, 0.34807592853875097, 0.3296058890286615, 0.34834740506995737, 0.3296058890286615, 0.3443874624841344, 0.3582038122242823, 0.34807592853875097, 0.35573505671528943, 0.37358777761377354, 0.37602561966470943, 0.36349315411130856, 0.3319319003442034, 0.3581877172880252, 0.3516948918297654, 0.3425572479360281, 0.3689238253426115, 0.34807592853875097, 0.3513729837370441, 0.35123744620513775, 0.36976605424570214, 0.3534724118895066, 0.3656455711361988, 0.36552482284950283, 0.3435783847553143, 0.3425572479360281, 0.36876096464618924, 0.3435783847553143, 0.372067361728404, 0.3722058175713312, 0.38247639490987734, 0.3703849625237802, 0.3753449583377547, 0.36016973320388823, 0.36390170709368475, 0.3763790257717668, 0.3693003071965125, 0.3722058175713312, 0.3598447879512476, 0.3767813160347283, 0.37641841311980484, 0.3778951819982763, 0.3722058175713312, 0.3722058175713312, 0.3757419651689517, 0.3767813160347283, 0.3767813160347283, 0.38050035636595375, 0.3722058175713312, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3296058890286615, 0.3425572479360281, 0.34807592853875097, 0.3296058890286615, 0.34332298067657535, 0.3513729837370441, 0.34332298067657535, 0.35123744620513775, 0.3296058890286615, 0.3296058890286615, 0.358479628150311, 0.36080845594204547, 0.34576883756964155, 0.3581877172880252, 0.34807592853875097, 0.3568391875688145, 0.3319319003442034, 0.3653285375440065, 0.3319319003442034, 0.34576883756964155, 0.358479628150311, 0.3587331421780602, 0.3689340498266667, 0.3513729837370441, 0.3587331421780602, 0.34332298067657535, 0.3598447879512476, 0.3425572479360281, 0.3586511803703075, 0.35679458277441956, 0.36016973320388823, 0.3703849625237802, 0.3722058175713312, 0.35800072278615447, 0.3699109223674845, 0.36016973320388823, 0.36349315411130856, 0.3722058175713312, 0.3689340498266667, 0.36016973320388823, 0.3722058175713312, 0.3722058175713312, 0.3722058175713312, 0.3722058175713312, 0.3722058175713312, 0.3722058175713312, 0.3722058175713312, 0.3722058175713312, 0.3722058175713312, 0.3722058175713312, 0.3296058890286615]
