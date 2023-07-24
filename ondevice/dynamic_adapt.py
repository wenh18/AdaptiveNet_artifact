import argparse
import copy
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.nn as nn
import torchvision.utils
from torch.utils.data import Subset
import numpy as np
import torch
import time
import torchvision
from mytimm.utils import *
from tools.evolution_finder import EvolutionFinder
import torchvision.transforms as transforms
import timm
from tools import mytools
from mytimm.models import create_model

parser = argparse.ArgumentParser(description='evolution finder', add_help=False)
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset')
# Model parameters
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')
parser.add_argument('--model_path', metavar='DIR',
                    help='path to trained model')
parser.add_argument("--warmupbatches", default=10, type=int)
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('-t', '--time_budget', type=float, default=0.9375-0.0625*2, metavar='N',
                    help='time budget factor (default: 0.9375-0.0625*2)')
parser.add_argument('--GPU', action='store_true', default=False,
                    help='Use GPU')
parser.add_argument('--pths_path',default='pths_for_device_torchvision/',  metavar='DIR',
                    help='path to trained model')
parser.add_argument('--subnet_path',default='./npy/ration1_population200_search6_datalen200.npy',  metavar='DIR',
                    help='subnet_path')
parser.add_argument('--subset_idxs',default='./npy/idxs.npy',  metavar='DIR',
                    help='subset_idxs_path')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')

def validate(model,loader,args):
    batch_time_m =  ()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    end = time.time()

    last_idx = len(loader) - 1
    total_time = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if args.GPU:
                input = input.cuda()
                target = target.cuda()
            t1 = time.time()
            for m in model:
                if m is not None:
                    input = m(input)
            output = input
            total_time += (time.time() - t1)
            if isinstance(output, (tuple, list)):
                output = output[0]

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            torch.cuda.synchronize()
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
    return top1_m.avg, total_time

def warmup(model, args, warmuptime=100):
    x = torch.randn(args.batch_size, 3, 224, 224)
    for _ in range(warmuptime):
        if args.GPU:
            xx = x.cuda()
        for m in model:
            if m is not None:
                xx = m(xx)

def deploy(subnets,loader_eval,validate,args):
        trade_off = mytools.get_tradeoff_subnets(subnets)
        current_subnet = trade_off[0][0]
        pre_subne = current_subnet
        current_subnet_idx = 0
        accs = []
        lats = []
        model = mytools.get_resnet_model_from_subnet([0 for i in range(len(current_subnet))], args.pths_path,use_presubnet=False,old_subnet=None,model=None)
        warmup(model,args)
        acc, latency=validate(model, loader_eval, args)
        print("validate_baseline acc:{}  latency:{}".format(acc,latency))
        time_budget = args.time_budget * latency
        model = mytools.get_resnet_model_from_subnet(current_subnet, args.pths_path,use_presubnet=False,old_subnet=None,model=None)
        while True:
            acc, latency=validate(model, loader_eval, args)
            accs.append(acc)
            lats.append(latency)
            print("{} acc:{} latency:{:.4f}  time_budget:{}".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),acc,latency,time_budget))
            
            if latency > time_budget:
                flag = -1
                for ind in range(current_subnet_idx-1,0,-1):
                    acc, tmp_latency=validate(model, loader_eval, args)
                    if tmp_latency < time_budget:
                        print('warning: latency increase, switch subnet:{} to {}'.format(current_subnet,trade_off[ind][0]))
                        old_subnet = current_subnet
                        current_subnet =  trade_off[ind][0]
                        model = mytools.get_resnet_model_from_subnet(current_subnet, args.pths_path,use_presubnet=True,old_subnet=old_subnet,model=model)
                        current_subnet_idx = ind
                        flag = 0
                        break
                if flag == -1:
                    old_subnet = current_subnet
                    current_subnet = trade_off[0][0]
                    model = mytools.get_resnet_model_from_subnet(current_subnet, args.pths_path,use_presubnet=True,old_subnet=old_subnet,model=model)
                    current_subnet_idx = 0
                    
            else:
                for ind in range(len(trade_off)-1,current_subnet_idx+1,-1):
                    acc, tmp_latency=validate(model, loader_eval, args)
                    if tmp_latency < time_budget - 0.001:
                        print('warning: latency decrease, switch subnet:{} to {}'.format(current_subnet,trade_off[ind][0]))
                        old_subnet = current_subnet
                        current_subnet =  trade_off[ind][0]
                        model = mytools.get_resnet_model_from_subnet(current_subnet, args.pths_path,use_presubnet=True,old_subnet=old_subnet,model=model)
                        current_subnet_idx = ind
                        break

def main():
    args = parser.parse_args()

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    	     std=[0.229, 0.224, 0.225])
    ])
    dataset_eval = torchvision.datasets.ImageFolder(root=args.data_dir,transform=data_transform)
    idxs = np.load(args.subset_idxs).tolist()[:500]
    eval_set = Subset(dataset_eval, idxs)
    loader_eval = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
    model = create_model(
    "resnet50",
    num_classes=args.num_classes,
    drop_rate=args.drop,
    drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
    drop_path_rate=args.drop_path,
    drop_block_rate=args.drop_block,
    global_pool=args.gp,
    bn_momentum=args.bn_momentum,
    bn_eps=args.bn_eps,
    scriptable=args.torchscript,
    checkpoint_path=args.initial_checkpoint)
    model.get_skip_blocks(more_options=False)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=True)
    model.cuda()
    mytools.extract_blocks_from_multimodel(model, savepath='pths_for_device_torchvision/', method='all')
    
    subnets = np.load(args.subnet_path,allow_pickle=True).tolist()
    deploy(subnets,loader_eval,validate, args)

if __name__ == '__main__':
    main()
