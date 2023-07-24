import argparse
import copy
import os
import random as rd

import logging
import torch.nn as nn
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
import numpy as np
import torch
import time
import torchvision
from mytimm.models import create_model
from mytimm.utils import *
# from train_efficientnet import load_multimodel
from evolution_baseline import EvolutionFinder
import torchvision.transforms as transforms
import timm
# import mytools
# from train_efficientnet import load_multimodel, load_to_MultiModel
from AdaptiveNet import oncloud
parser = argparse.ArgumentParser(description='evolution finder', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
from prebuilt_latency_table import LATENCY_TABLE

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
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
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
parser.add_argument('--GPU', action='store_true', default=False,
                    help='Use GPU')
parser.add_argument("--log_interval", default=200, type=int)
parser.add_argument("--warmupbatches", default=10, type=int)
parser.add_argument('--pths_path', metavar='DIR',
                    help='path to trained model')
parser.add_argument('--slim', action='store_true', default=False)
parser.add_argument('--use_subset', action='store_true', default=False)
parser.add_argument("--batch_size_for_lat", default=1, type=int)
parser.add_argument('--pruned', action='store_true', default=False)

parser.add_argument('--data_aware', action='store_true', default=False)
parser.add_argument('--randomlysample', action='store_true', default=False)
parser.add_argument('--test_effi', action='store_true', default=False)
parser.add_argument('--dirichlet_alpha', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--test_device',  default='jetson_nano', type=str,
                    help='jetson_nano, xiaomi, 3090')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_logger = logging.getLogger('train')

def load_multimodel(model, path):
    # model.load_state_dict(torch.load(path), strict=True)
    state = {}
    model_dict = model.state_dict()
    pretrained_model = torch.load(path, map_location=torch.device('cpu'))
    for k, v in pretrained_model.items():
        # print(k)
        key = k[7:] if "module" in k else k
        state[key] = v
    model_dict.update(state)
    model.load_state_dict(model_dict)
    for (name, parameter) in model.named_parameters():
        # print(name)
        name_list = name.split('.')
        if name_list[0] == "multiblocks" and name_list[2] == "0":
            print(";;;", name)
            parameter.requires_grad = False
        if name_list[0] == "conv_stem" or name_list[0] == "bn1" or name_list[0] == "bn2" or name_list[0] == "conv_head":
            print(";;;", name)
            parameter.requires_grad = False
        if "classifier" in name:
            parameter.requires_grad = False
            print(";;;", name)
    return model

def test_lat(block, input, test_times):
    x = torch.rand(input.shape).cuda()
    block.cuda()
    lats = []
    for i in range(test_times):
        t1 = time.time()
        y = block(x)
        torch.cuda.synchronize()
        t2 = time.time() - t1
        if i > 200:
            lats.append(t2)
        del y
    del x
    return np.mean(lats)

def get_resnet_lats(model, batchsize, test_times=1000):
    print("started testing latencys")
    model.eval()
    model.cuda()
    x = torch.rand(batchsize, 3, 224, 224).cuda()  # 224
    lats = []
    layers = [model.conv1, model.bn1, model.act1, model.maxpool]
    former_layers = nn.Sequential(*layers)
    former_layers.cuda()
    # x.cuda()
    lats.append(test_lat(former_layers, x, test_times))
    x = former_layers(x)
    for blockidx in range(len(model.multiblocks)):
        print("testing latency for the",blockidx, "block")
        lats.append([])
        for choiceidx in range(len(model.multiblocks[blockidx])):
            block = copy.deepcopy(model.multiblocks[blockidx][choiceidx])
            lats[-1].append(test_lat(block, x, test_times))
        x = model.multiblocks[blockidx][0](x)
    f_layers = [model.global_pool, model.fc]
    latter_layers = nn.Sequential(*f_layers)
    lats.append(test_lat(latter_layers, x, test_times))
    return lats

def get_mbv_lats(model, batchsize, test_times=1000):
    model.eval()
    model.cuda()
    x = torch.rand(batchsize, 3, 288, 288).cuda()  # 224 384 416
    lats = []
    layers = [model.conv_stem, model.bn1, model.act1]
    former_layers = nn.Sequential(*layers)
    # former_layers.cuda()
    # x.cuda()
    lats.append(test_lat(former_layers, x, test_times))
    x = former_layers(x)
    del former_layers
    for blockidx in range(len(model.multiblocks)):
        lats.append([])
        for choiceidx in range(len(model.multiblocks[blockidx])):
            block = copy.deepcopy(model.multiblocks[blockidx][choiceidx])
            lats[-1].append(test_lat(block, x, test_times))
        x = model.multiblocks[blockidx][0](x)
    f_layers = [model.conv_head, model.bn2, model.act2, model.global_pool, model.classifier]
    latter_layers = nn.Sequential(*f_layers)
    lats.append(test_lat(latter_layers, x, test_times))
    return lats

def validate(model, loader, subnet, args, loss_fn, lats):
    print(subnet)
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    # model = copy.deepcopy(originalmodel)
    # model.adjust_multiblocks_to_subnet(subnet)
    # model = mytools.get_resnet_model_from_subnet(originalmodel, subnet, args.pths_path)
    lat = lats[0]
    # print(subnet)
    for blockidx in range(len(subnet)):
        if subnet[blockidx] != 99:
            # print(blockidx, subnet[blockidx])
            lat += lats[blockidx+1][subnet[blockidx]]
    lat += lats[-1]
    if args.GPU:
        model.cuda()
    model.eval()
    end = time.time()
    # import pdb;pdb.set_trace()
    last_idx = len(loader) - 1
    total_time = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # print(batch_idx)
            # print(batch_idx, input.shape, target.shape)
            last_batch = batch_idx == last_idx
            if args.GPU:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            # with amp_autocast():
            t1 = time.time()
            # if teachermodel:
            #     output = model(input)
            # else:
            #     output = model(input, subnet)
            output = model(input, subnet)
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
                # if stage == 3:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            # torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            # if stage == 3:
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            # print("over")
            # if last_batch:#or (batch_idx % args.log_interval == 0):
            #     # print(subnet, subnetchoice)
            #     log_name = 'Test'
            #     print(log_name, "batchidx:", batch_idx, "latency", batch_time_m.avg,
            #             "loss", losses_m.avg, "acc", top1_m.avg)
    # total_time /= (last_idx + 1 - args.warmupbatches)
    # del model
    print(top1_m.avg, lat, subnet)
    return top1_m.avg, lat

def warmup(model, args, warmuptime, teachermodel=False, subnet=None):
    x = torch.randn(args.batch_size, 3, 224, 224)
    if args.GPU:
        x.cuda()
    for _ in range(warmuptime):
        if teachermodel:
            output = model(x)
        else:
            output = model(x, subnet)

def get_output_dir(args):
    if 'resnet50' in args.model:
        model_name = 'res50'
    if 'resnet101' in args.model:
        model_name = 'res101'
    if 'mobilenet' in args.model:
        model_name = 'mbv2'
    if 'jetson' in args.test_device:
        device = 'jetson'
    if '3090' in args.test_device:
        device = 'nvidia_3090'
    if 'xiaomi' in args.test_device:
        device = 'xiaomi'
    return 'visualization/searched/'+device + '_' + model_name + '_info.yaml'

def main():
    args = parser.parse_args()
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    args.world_size = 1
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,)
    if 'resnet' in args.model:
        model.get_skip_blocks(more_options=False)
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules = oncloud.prune_model(teachermodel, [0.2, 0.4])
            model.get_pruned_module(prune_modules)
            del teachermodel
        lats = LATENCY_TABLE[args.test_device][args.model]

    if 'mobilenetv2' in args.model:
        model.get_skip_blocks()
        if args.pruned:
            teachermodel = timm.create_model(args.model, pretrained=True)
            prune_modules = oncloud.prune_mbv2(teachermodel, [0.25, 0.5])
            model.get_pruned_module(prune_modules)
            del teachermodel
        lats = LATENCY_TABLE[args.test_device][args.model]

    if 'efficientnet' in args.model:
        model.get_skip_blocks()
        load_multimodel(model, args.model_path)
        lats = LATENCY_TABLE[args.model]
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=True)

    loss_fn = nn.CrossEntropyLoss()
    if args.GPU:
        # originalmodel.cuda()
        loss_fn.cuda()
    if 'v2_rw_m' in args.model:
        print('-------------v2m--------------')
        imagesize=416
    elif 'v2_rw_s' in args.model:
        print('-------------v2s--------------')
        imagesize = 384
    elif 'v2_rw_t' in args.model:
        print('-------------v2t--------------')
        imagesize = 288
    else:
        imagesize = 224
    data_transform = transforms.Compose([
        transforms.Resize(imagesize),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset_eval = torchvision.datasets.ImageFolder(root=args.data_dir,transform=data_transform)

    if args.test_effi:
        loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=16)
        baseline_subnet = model.generate_random_subnet()
        baseline_subnet = [0 for _ in range(len(baseline_subnet))]
        baseline_acc, baseline_latency = validate(model, loader_eval, baseline_subnet, args, loss_fn, lats)
        accs, latencys = [baseline_acc], [baseline_latency]
        for i in range(1, len(infos)):
            subnet = infos[i][1]
            acc, lat = validate(model, loader_eval, subnet, args, loss_fn, lats)
            accs.append(acc)
            latencys.append(lat)
        print(accs, ',', latencys)
        exit(0)

    modeltyp = 'mbv2' if 'mobilenetv2' in args.model else 'resnet'
    if 'mobilenetv2_120d' in args.model:
        modeltyp='mbv2d'
    if 'v2_rw_s' in args.model:
        modeltyp='effi_s'
    print('searching for', modeltyp)


    if args.data_aware and args.randomlysample:
        for alpha in [args.dirichlet_alpha]:
            # from niid_data import getdirichletprobs, getdataidxs
            # probs=getdirichletprobs(alpha)
            # print(probs)
            # sampler_idxs = getdataidxs(probs, False, 5000)
            sampler_idxs = np.load('npy/dirichlet_randomlysample_'+str(int(alpha*1000))+'.npy',allow_pickle=True)
            # np.save('npy/dirichlet_randomlysample_'+str(int(alpha*1000))+'.npy', sampler_idxs)
            print("finished building sampler for alpha:", alpha)
            test_set = Subset(dataset_eval, sampler_idxs)
            loader_test = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=16)
            original_subnet = model.generate_random_subnet()
            original_subnet = [0 for _ in range(len(original_subnet))]
            baseline_acc, baseline_latency = validate(model, loader_test, original_subnet, args, loss_fn, lats)
            modeltyp = 'effi_s' if args.model == 'efficientnetv2_rw_s' else 'effi_m' # effi is not mobilenetv2 type
            print('searching for', modeltyp)#, 'results sampled on iid data:', best_infos)
            from evolution_baseline import generate_subnets
            subnets = generate_subnets(3000, len(original_subnet), type=modeltyp, lessskip=True)
            subnets_name = 'npy/effim/'+args.model +str(int(alpha*1000))+ '_dirichlet_subnets.npy'
            np.save(subnets_name, subnets)
            niid_accs, niid_lats = [baseline_acc], [baseline_latency]
            for subnetidx in range(len(subnets)):
                acc, lat = validate(model, loader_test, subnets[subnetidx], args, loss_fn, lats)
                niid_accs.append(acc)
                niid_lats.append(lat)
                if subnetidx % 200 == 0:
                    print('*********************************', subnetidx, '***********************************')
                    print(niid_accs, ',', niid_lats)
            np.save('npy/effim/data_aware'+str(int(1000*alpha))+args.model + '_randomlysample_results.npy', [niid_accs, niid_lats])
        exit(0)

    elif args.data_aware:
        results = []
        search = torchvision.datasets.ImageFolder(root='../datasets/imagenet/SearchForDevice3k/',transform=data_transform)
        test = torchvision.datasets.ImageFolder(root='../datasets/imagenet/ValForDevice3k/',transform=data_transform)
        # baseline_latency = 0.07170519512496396
        for alpha in [args.dirichlet_alpha]:
            results.append([])
            from niid_data import getdirichletprobs, getdataidxs
            # sampler_idxs = get_niid_data(args.data_dir, n_clients=10, DIRICHLET_ALPHA=alpha)
            probs=getdirichletprobs(alpha)
            print(probs)
            sampler_idxs_search=getdataidxs(probs, True, 5000)
            print(sampler_idxs_search[-1])
            print(probs)
            sampler_idxs = getdataidxs(probs, False, 5000)
            print(sampler_idxs[-1])
            np.save('npy/dirichlet_search/dirichlet_search_'+str(int(alpha*1000))+'.npy', sampler_idxs_search)
            np.save('npy/dirichlet_search/dirichlet_test_' + str(int(alpha * 1000)) + '.npy', sampler_idxs)
            print("finished building sampler for alpha:", alpha)
            # for i in range(1):
                # rd.shuffle(sampler_idxs[i])
                # search_idxs = sampler_idxs[i][:int(0.5*len(sampler_idxs[i]))]
                # test_idxs = sampler_idxs[i][int(0.5*len(sampler_idxs[i])):]
                # print("------------------------------", len(search_idxs), len(test_idxs), "----------------------------------------")
                # eval_set = Subset(dataset_eval, search_idxs)
                # test_set = Subset(dataset_eval, test_idxs)
            search_set = Subset(search, sampler_idxs_search)
            test_set = Subset(test, sampler_idxs)
            loader_search = torch.utils.data.DataLoader(search_set, batch_size=args.batch_size, shuffle=False, num_workers=16)
            loader_test = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=16)

            original_subnet = model.generate_random_subnet()
            original_subnet = [0 for _ in range(len(original_subnet))]
            baseline_acc, baseline_latency = validate(model, loader_test, original_subnet, args, loss_fn, lats)

            modeltyp = 'effi_s' #'mbv2' if 'mobilenetv2' in args.model else 'resnet'  # effi is not mobilenetv2 type
                # because it elasticizes the first conv layer, yet mbv2 didnt
            print('searching for', modeltyp)#, 'results sampled on iid data:', best_infos)
            for ratio in range(10):
                print("--------------", 0.95-0.05 * ratio, "---------------")
                finder = EvolutionFinder(batch_size=args.batch_size, branch_choices=model.block_choices,
                                         time_budget=(0.95-0.05 * ratio) * baseline_latency, searching_times=6,
                                         lats=lats, modeltype=modeltyp, pruned=args.pruned)
                _, best_info = finder.evolution_search(model, validate, loader_search, args, loss_fn)
                test_acc, test_lat = validate(model, loader_test, best_info[1], args, loss_fn, lats)
                results[-1].append([best_info, (test_acc, test_lat), (baseline_acc, baseline_latency)])
                print(results)
            np.save('npy/data_aware'+str(int(1000*alpha)) + '_results.npy', results)
        exit(0)


    dataset_eval = torchvision.datasets.ImageFolder(root='../datasets/imagenet/SearchForDevice3k',transform=data_transform)
    loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=16)
    dataset_test = torchvision.datasets.ImageFolder(root='../datasets/imagenet/ValForDevice3k/',transform=data_transform)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=16)
    print('number of test data:', len(loader_eval)*args.batch_size)

    baseline_subnet = model.generate_random_subnet()
    baseline_subnet = [0 for _ in range(len(baseline_subnet))]


    baseline_acc, baseline_latency = validate(model, loader_eval, baseline_subnet, args, loss_fn, lats)
    model_len = len(baseline_subnet)
    pruned_points = []
    for i in range(model_len):
        for j in range(len(model.block_choices[i])):
            if model.block_choices[i][j] < 0:
                pruned_points.append(i)
                break

    if 'resnet' in args.model:
        pruned_points = [i for i in range(len(baseline_subnet))]
    print("pruned_points:", pruned_points)
    # originalmodel.cpu()
    best_infos = [(baseline_acc, baseline_latency)]
    for ratio in range(10):
        print("--------------", 0.95-0.05*ratio, "---------------")
        finder = EvolutionFinder(batch_size=args.batch_size, branch_choices=model.block_choices,
                                 time_budget=(0.95-0.05*ratio)*baseline_latency, searching_times=5, lats=lats, modeltype=modeltyp, pruned=args.pruned, pruned_points=pruned_points, propulation_size=200)

        _, best_info = finder.evolution_search(model, validate, loader_eval, args, loss_fn)
        test_acc, test_lat = validate(model, loader_test, best_info[1], args, loss_fn, lats)
        best_infos.append([best_info, test_acc, test_lat])
        print(best_infos)
    print(best_infos)
    import yaml
    output_dir = get_output_dir(args)
    with open(output_dir, 'w') as f:
        yaml.dump(best_infos, f)
    # np.save('visualization/searched/'+ args.model + args.test_device + '_results.npy', best_infos)

if __name__ == '__main__':
    main()
