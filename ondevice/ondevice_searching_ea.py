import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import logging
import torch.nn as nn
import torchvision.utils
from torch.utils.data import Subset
import numpy as np
import torch
import time
import torchvision
from mytimm.models import create_model
from mytimm.utils import *
from tools.evolution_finder import EvolutionFinder
import torchvision.transforms as transforms
import timm
from tools import mytools
from tools import global_var

parser = argparse.ArgumentParser(description='evolution finder', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Model parameters
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
#parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   # help='input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')

parser.add_argument('--model_path', metavar='DIR',
                    help='path to trained model')

parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--GPU', action='store_true', default=True,
                    help='Use GPU')
parser.add_argument("--log_interval", default=200, type=int)
parser.add_argument("--warmupbatches", default=10, type=int)
parser.add_argument('--pths_path', metavar='DIR',
                    help='path to trained model')
parser.add_argument('--slim', action='store_true', default=False)
parser.add_argument('--use_subset', action='store_true', default=False)
parser.add_argument("--batch_size_for_lat", default=4, type=int)
parser.add_argument('--pruned', action='store_true', default=False)
parser.add_argument('--save_path', default='./npy/batch-', type=str)
parser.add_argument('--baseline_save_path', default='./npy/baseline-batch-', type=str)
parser.add_argument("--num_workers", default=8, type=int)


parser.add_argument("--data_len", default=500, type=int)
parser.add_argument("--pre_len", default=3, type=int) 
parser.add_argument("--searching_times", default=1, type=int)
parser.add_argument("--population_size", default=100, type=int)

parser.add_argument("--load_times", default=2, type=int) 
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--baseline_load_times", default=1, type=int)
parser.add_argument("--baseline_batch_size", default=500, type=int)
parser.add_argument('--method', default='AdaptiveNet',type=str)

def test_lat(block, input, test_times):
    x = torch.rand(input.shape).cuda()
    lats = []
    for i in range(test_times):
        t1 = time.time()
        _ = block(x)
        torch.cuda.synchronize()
        t2 = time.time() - t1
        if i > 200:
            lats.append(t2)
    return np.mean(lats)

def get_resnet_lats(model, batchsize, test_times=500):
    model.eval()
    model.cuda()
    x = torch.rand(batchsize, 3, 224, 224).cuda()
    lats = []
    layers = [model.conv1, model.bn1, model.act1, model.maxpool]
    former_layers = nn.Sequential(*layers)
    former_layers.cuda()
    lats.append(test_lat(former_layers, x, test_times))
    x = former_layers(x)
    for blockidx in range(len(model.multiblocks)):
        lats.append([])
        for choiceidx in range(len(model.multiblocks[blockidx])):
            lats[-1].append(test_lat(model.multiblocks[blockidx][choiceidx], x, test_times))
        x = model.multiblocks[blockidx][0](x)
    f_layers = [model.global_pool, model.fc]
    latter_layers = nn.Sequential(*f_layers)
    lats.append(test_lat(latter_layers, x, test_times))
    return lats

def get_mbv_lats(model, batchsize, test_times=1000):
    model.eval()
    model.cuda()
    x = torch.rand(batchsize, 3, 224, 224).cuda()
    lats = []
    layers = [model.conv_stem, model.bn1, model.act1]
    former_layers = nn.Sequential(*layers)
    former_layers.cuda()
    lats.append(test_lat(former_layers, x, test_times))
    x = former_layers(x)
    for blockidx in range(len(model.multiblocks)):
        lats.append([])
        for choiceidx in range(len(model.multiblocks[blockidx])):
            lats[-1].append(test_lat(model.multiblocks[blockidx][choiceidx], x, test_times))
        x = model.multiblocks[blockidx][0](x)
    f_layers = [model.conv_head, model.bn2, model.act2, model.global_pool, model.classifier]
    latter_layers = nn.Sequential(*f_layers)
    lats.append(test_lat(latter_layers, x, test_times))
    return lats

def validate_baseline(model, loader, subnet, args, loss_fn, lats):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    lat = lats[0]
    for blockidx in range(len(subnet)):
        if subnet[blockidx] != 99:
            lat += lats[blockidx+1][subnet[blockidx]]
    lat += lats[-1]
    if args.GPU:
        model.cuda()
    model.eval()
    with torch.no_grad():
       for batch_idx, (input, target) in enumerate(loader):
            if args.GPU:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            # with amp_autocast():
            output = model(input, subnet,batch_idx=batch_idx,infer_type=0)
            torch.cuda.synchronize()

            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
    return top1_m.avg, lat

def validate(model, loader, subnet, args, loss_fn, lats,infer_type=2):
    validate_time = time.time()
    get_subnet_time = 0
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    lat = lats[0]
    for blockidx in range(len(subnet)):
        if subnet[blockidx] != 99:
            lat += lats[blockidx+1][subnet[blockidx]]
    lat += lats[-1]
    if args.GPU:
        model.cuda()
    model.eval()
    check_validated_feature_time = time.time()
    if infer_type == 2:
        model.check_validated_feature()
    check_validated_feature_time = time.time()-check_validated_feature_time
    end = time.time()
    last_idx = len(loader) - 1
    total_time = 0
    infer_time = 0 
    infer_after_time = 0
    to_gpu_time = 0
    with torch.no_grad():
        for batch_idx in range(len(loader)):
            input = loader[batch_idx][0]
            target = loader[batch_idx][1]
            last_batch = batch_idx == last_idx
            # if args.GPU:
            #     input = input.cuda()
            #     target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            # with amp_autocast():
            s2 = time.time()
            t1 = time.time()
            output = model(input, subnet,batch_idx=batch_idx,infer_type=infer_type)
            torch.cuda.synchronize()
            infer_time += time.time()-s2
            s3 = time.time()
            if batch_idx >= args.warmupbatches:
                total_time += (time.time() - t1)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
    update_validated_feature_time  = time.time()
    if infer_type == 2:
        model.update_validated_feature()
    update_validated_feature_time = time.time()-update_validated_feature_time
    validate_time = time.time()-validate_time
    return top1_m.sum, top1_m.count, lat

def warmup(model, args, warmuptime, teachermodel=False, subnet=None):
    x = torch.randn(args.batch_size, 3, 224, 224)
    if args.GPU:
        x.cuda()
    for _ in range(warmuptime):
        if teachermodel:
            output = model(x)
        else:
            output = model(x, subnet)


def main():
    args = parser.parse_args()
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    args.world_size = 1
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=1000,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    if 'resnet' in args.model:
        model.get_skip_blocks(more_options=False)
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules = mytools.prune_model(teachermodel, [0.2, 0.4])
            model.get_pruned_module(prune_modules)
            del teachermodel
    if 'mobilenetv2' in args.model:
        model.get_multi_blocks()
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules = mytools.prune_mbv2(teachermodel, [0.25, 0.5])
            model.get_pruned_module(prune_modules)
            del teachermodel

    global_var._init()
    global_var.set_value('validated_feature', set())
    global_var.set_value('need_save_feature', set())
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=True)
    model.cuda()
    if args.method == "AdaptiveNet":
        mytools.load_data(model,args.model,args.data_dir,args.save_path, method = args.method, load_times=args.load_times,batch_size=args.batch_size,data_len=args.data_len)
    elif args.method == "BaseLine0":
        mytools.load_data(model,args.model,args.data_dir,args.baseline_save_path, method = args.method, load_times=args.load_times,batch_size=args.batch_size,data_len=args.data_len)
    
    if args.slim:
        model.adjust_channels()
        model.get_multi_blocks()
    elif 'mobilenetv2' in args.model:
        layer_lens = []
        for layeridx in range(len(model.multiblocks)):
            layerlen = len(model.multiblocks[layeridx])
            for blockidx in range(len(model.multiblocks[layeridx])): # note that we did not prune the last block
                layer_lens.append(layerlen)
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules, prune_checkpoint, prune_block_nums = mytools.prune_model_v2(teachermodel, [0.25, 0.5])
            model.get_pruned_module(prune_modules, prune_checkpoint, prune_block_nums)
            del teachermodel

    if "resnet" in args.model:
        lats = get_resnet_lats(model, batchsize=args.batch_size_for_lat)
    else:
        lats = get_mbv_lats(model, batchsize=args.batch_size_for_lat)
    loss_fn = nn.CrossEntropyLoss()
    if args.GPU:
        loss_fn.cuda()

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset_eval = torchvision.datasets.ImageFolder(root=args.data_dir,transform=data_transform)
    idxs = np.load('./npy/idxs.npy').tolist()[:args.data_len]
    eval_set = Subset(dataset_eval, idxs)
    loader_eval = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    baseline_subnet = model.generate_random_subnet()
    baseline_subnet = [0 for _ in range(len(baseline_subnet))]
    acc, baseline_latency = validate_baseline(model, loader_eval, baseline_subnet, args, loss_fn, lats)
    print("method : {} population_size : {}  validate_baseline acc:{}  latency:{}".format(args.method, args.population_size, acc, baseline_latency))
    t1 = time.time()
    lens = layer_lens if "mobilenetv2" in args.model else None
    finder = EvolutionFinder(batch_size=args.batch_size, population_size=args.population_size, branch_choices=model.block_choices, time_budget=(0.9375-0.0625*2)*baseline_latency, searching_times=args.searching_times, lats=lats, model_lens=lens)
    if args.method == "AdaptiveNet":
        _, best_info = finder.evolution_search(model, validate, args, loss_fn)
    elif args.method == "BaseLine0":
        _, best_info = finder.evolution_search_baseline1(model, validate, args, loss_fn)
    print("best result:{}".format(best_info))
    print("total_time:{}s\n\n".format(time.time()-t1))


if __name__ == '__main__':
    main()