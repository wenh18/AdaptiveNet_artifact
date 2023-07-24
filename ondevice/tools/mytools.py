import torchvision
from torch.utils.data import Subset
import numpy as np
import copy
import torch
import random
import math
import pynvml
from itertools import combinations
import math
from torch.utils.data.sampler import BatchSampler
from torch.utils.mobile_optimizer import optimize_for_mobile
import os,sys,time
import torchvision.datasets as datasets
from torchvision import transforms
import tools.global_var


def extract_blocks_from_multimodel(multimodel, savepath, method='all', model_type='resnet'):
    if not os.path.exists(savepath):
        os.makedirs(savepath) 
    multimodel = multimodel.eval()
    if method == 'pth':
        torch.save(multimodel.conv1.state_dict(), savepath + "conv1.pth")
        torch.save(multimodel.bn1.state_dict(), savepath + "bn1.pth")
        torch.save(multimodel.fc.state_dict(), savepath + "fc.pth")
        for block_idx in range(len(multimodel.multiblocks)):
            for block_choice in range(len(multimodel.multiblocks[block_idx])):
                name = savepath + "block-{}-{}.pth".format(block_idx, block_choice)
                torch.save(multimodel.multiblocks[block_idx][block_choice].state_dict(), name)
    elif method == 'jit':
        example = torch.rand(1, 3, 224, 224).cuda()
        if model_type == 'resnet':
            formerlayers = torch.nn.Sequential(multimodel.conv1, multimodel.bn1, multimodel.relu, multimodel.maxpool)

            traced_script_module = torch.jit.script(formerlayers, example)
            traced_script_module_optimized = optimize_for_mobile(traced_script_module)
            traced_script_module_optimized._save_for_lite_interpreter(savepath + "formerlayers.pt")

            x = multimodel.conv1(example)
            x = multimodel.bn1(x)
            x = multimodel.relu(x)
            example1 = multimodel.maxpool(x)

            for blockidx in range(len(multimodel.multiblocks)):
                for blockchoice in range(len(multimodel.multiblocks[blockidx])):
                    block = multimodel.multiblocks[blockidx][blockchoice]
                    traced_script_module = torch.jit.script(block, example1)
                    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
                    traced_script_module_optimized._save_for_lite_interpreter(savepath + "block-{}-{}.pt".format(blockidx, blockchoice))

                example1 = multimodel.multiblocks[blockidx][0](example1)

            latterlayers = torch.nn.Sequential(multimodel.global_pool, multimodel.fc)

            traced_script_module = torch.jit.script(latterlayers, example1)
            traced_script_module_optimized = optimize_for_mobile(traced_script_module)
            traced_script_module_optimized._save_for_lite_interpreter(savepath + "latterlayers.pt")
    elif method == 'all':
        formerlayers = torch.nn.Sequential(multimodel.conv1, multimodel.bn1, multimodel.relu, multimodel.maxpool)
        latterlayers = torch.nn.Sequential(multimodel.global_pool, multimodel.fc)
        torch.save(formerlayers, savepath + "formerlayers.pth")
        torch.save(latterlayers, savepath + "latterlayers.pth")
        for block_idx in range(len(multimodel.multiblocks)):
            for block_choice in range(len(multimodel.multiblocks[block_idx])):
                name = savepath + "block-{}-{}.pth".format(block_idx, block_choice)
                torch.save(multimodel.multiblocks[block_idx][block_choice], name)

def get_data(args):
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader

def get_tradeoff_subnets(subnets, iter_num=0):
    #[1, 99, 1, 99, 1, 99, 1, 99, 2, 99, 99, 2, 99, 99, 0, 0],0.006877277208411175,65.60000610351562]
    xs = []
    ys = []
    subnet = [] 
    subnets.sort(key=lambda x:x[1])
    for i in range(len(subnets)):
        subnet.append(subnets[i][0])
        xs.append(subnets[i][1])
        ys.append(subnets[i][2])
    extreme_idxs = [0]
    max_acc = -1
    for idx in range(len(xs)):
        if ys[idx] > max_acc:
            extreme_idxs.append(idx)
            max_acc = ys[idx]
    for _ in range(iter_num):
        cp_extreme_idxs = copy.deepcopy(extreme_idxs)
        for idx in range(1, len(extreme_idxs) - 1):
            if ys[extreme_idxs[idx]] <= ys[extreme_idxs[idx-1]] and ys[extreme_idxs[idx]] <= ys[extreme_idxs[idx+1]]:
                cp_extreme_idxs[idx] = -1
        while -1 in cp_extreme_idxs:
            cp_extreme_idxs.remove(-1)
        extreme_idxs = cp_extreme_idxs
    extreme_idxs.append(len(xs)-1)
    extreme_idxs.sort()
    result = []
    for i in extreme_idxs:
        result.append([subnet[i],xs[i],ys[i]])
    return result[3:]

def get_resnet_model_from_subnet(cur_subnet, blockspath,use_presubnet=False,old_subnet=None,model=None):
    if use_presubnet:
        for blockidx in range(len(old_subnet)):
            if old_subnet[blockidx] != cur_subnet[blockidx]:
                if cur_subnet[blockidx] == 99:
                    model[blockidx+1] = None
                else:
                    blockpath = blockspath + "block-{}-{}.pth".format(blockidx, cur_subnet[blockidx])
                    model[blockidx+1] = torch.load(blockpath, map_location=torch.device('cuda'))
    else:
        model = []
        model.append(torch.load(blockspath+'formerlayers.pth',map_location=torch.device('cuda')))
        for blockidx in range(len(cur_subnet)):
            if cur_subnet[blockidx] != 99:
                blockpath = blockspath+ "block-{}-{}.pth".format(blockidx, cur_subnet[blockidx])
                model.append(torch.load(blockpath, map_location=torch.device('cuda')).eval())
            else:
                model.append(None)
        model.append(torch.load(blockspath+'latterlayers.pth', map_location=torch.device('cuda')).eval())
    return model

def get_resnet_former_and_latter_layers(originalmodel, blockpath):
    '''
        for validating subnets on device only, do not use this method in other processes
    '''
    model_dict = originalmodel.state_dict()
    state = {}
    bn1_path = blockpath + "bn1.pth"
    conv1_path = blockpath + "conv1.pth"
    fc_path = blockpath + "fc.pth"

    conv1_weights = torch.load(conv1_path, map_location=torch.device('cpu'))
    for k, v in conv1_weights.items():
        key = "conv1." + k
        state[key] = v
    bn1_weights = torch.load(bn1_path, map_location=torch.device('cpu'))
    for k, v in bn1_weights.items():
        key = "bn1." + k
        state[key] = v
    fc_weights = torch.load(fc_path, map_location=torch.device('cpu'))
    for k, v in fc_weights.items():
        key = "fc." + k
        state[key] = v

    model_dict.update(state)
    originalmodel.load_state_dict(model_dict)
    return originalmodel

def get_model_size(model: torch.nn.Module, return_MB=False):
    tmp_model_file_path = 'tmp.model'
    torch.save(model, tmp_model_file_path)
    model_size = os.path.getsize(tmp_model_file_path)
    os.remove(tmp_model_file_path)
    if return_MB:
        model_size /= 1024 ** 2
    return model_size

def generate_subnets(sample_num=100, model_len=16, pruned=False, prune_points=None, type='mobilenetv2_100'):
    '''
    suggested:sample_num >= 700
    '''
    skip_rate = [0.1 + 0.1*i for i in range(10)]
    subnets = []
    for _ in range(50):
        for i in range(len(skip_rate)):
            for distill_next_rate in [1., 0.8, 0.6, 0.5, 0.4, 0.2, 0.]:
                for test_times in range(10):
                    blockidx = 0
                    subnet = []
                    while blockidx < model_len:
                        if pruned:
                            choices = [-2,-1,0] if blockidx in prune_points else [0]
                        else:
                            choices = [0]
                        # dealing with resnets please use "choices = [0] if not pruned else [-2, -1, 0]"
                        if type == 'mobilenetv2_100':
                            if 0 < blockidx < model_len - 1:  # for resnet do not 0 <, for mbv2, its 0<
                                choices.append(1)  # distill next one
                            if 1 < blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d, its 0 <, for mbv2_100 or 140, its 1 <
                                choices.append(2)  # distill next two
                            if 1 < blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d,  for mbv2_100 or 140, its 1 <
                                skipnext_prob = skip_rate[i] * distill_next_rate
                                skip_next_next_prob = skip_rate[i] * (1 - distill_next_rate)
                                if len(choices) == 3:
                                    probs = [(1 - skip_rate[i]), skipnext_prob, skip_next_next_prob]
                                else:
                                    probs = [skipnext_prob / 2., skip_next_next_prob / 2., 1 - skip_rate[i], skipnext_prob / 2., skip_next_next_prob / 2.]
                                choice = np.random.choice(choices, p=probs)
                            else:
                                choice = np.random.choice(choices)

                        elif type == 'mobilenetv2_100' or type == 'effi_s':
                            if 0 < blockidx < model_len - 1:  # for resnet do not 0 <, for mbv2, its 0<
                                choices.append(1)  # distill next one
                            if 0 < blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d, its 0 <, for mbv2_100 or 140, its 1 <
                                choices.append(2)  # distill next two
                            if 0 < blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d,  for mbv2_100 or 140, its 1 <
                                skipnext_prob = skip_rate[i] * distill_next_rate
                                skip_next_next_prob = skip_rate[i] * (1 - distill_next_rate)
                                if len(choices) == 3:
                                    probs = [(1 - skip_rate[i]), skipnext_prob, skip_next_next_prob]
                                else:
                                    probs = [skipnext_prob / 2., skip_next_next_prob / 2., 1 - skip_rate[i], skipnext_prob / 2., skip_next_next_prob / 2.]
                                choice = np.random.choice(choices, p=probs)
                            else:
                                choice = np.random.choice(choices)
                        else:
                            if blockidx < model_len - 1:  # for resnet do not 0 <, for mbv2, its 0<
                                choices.append(1)  # distill next one
                            if blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d, its 0 <, for mbv2_100 or 140, its 1 <
                                choices.append(2)  # distill next two
                            if blockidx < model_len - 2:  # for resnet do not 1 <, for mbv2_120d,  for mbv2_100 or 140, its 1 <
                                skipnext_prob = skip_rate[i] * distill_next_rate
                                skip_next_next_prob = skip_rate[i] * (1 - distill_next_rate)
                                if len(choices) == 3:
                                    probs = [(1 - skip_rate[i]), skipnext_prob, skip_next_next_prob]
                                else:
                                    probs = [(1 - skip_rate[i])/3, (1 - skip_rate[i])/3, (1 - skip_rate[i])/3, skipnext_prob, skip_next_next_prob]
                                choice = np.random.choice(choices, p=probs)
                            else:
                                choice = np.random.choice(choices)
                        if choice == 1:
                            subnet += [1, 99]
                            blockidx += 2
                        elif choice == 2:
                            subnet += [2, 99, 99]
                            blockidx += 3
                        else:
                            subnet.append(choice)
                            blockidx += 1
                    subnets.append(subnet)
    new_subnets = []
    for subnetidx in range(len(subnets)):
        if subnets[subnetidx] not in new_subnets:
            new_subnets.append(subnets[subnetidx])
    np.random.shuffle(new_subnets)
    return new_subnets[:sample_num]

def get_gpu_freememory(gpu_id): # return gpu(torch.device) with largest free memory.
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return int(mem_info.free/1024/1024)

def get_batchsize(model,input_size,device_id,alpha=1.1):
    baseline_subnet = [0 for _ in range(len(model.multiblocks))]
    with torch.no_grad():
        allocated1 = torch.cuda.max_memory_allocated() / (1024 ** 2.) 
        x = torch.randn(input_size).float().to(device_id)
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        for i in range(len(model.multiblocks)):
            x = model.multiblocks[i][0](x)
        allocated2 = torch.cuda.max_memory_allocated() / (1024 ** 2.)
        batch_size = int(get_gpu_freememory(device_id) / ((allocated2-allocated1)*alpha))
    return batch_size if batch_size % 2 == 0 else batch_size - 1

def get_max_feature_size(model,input_size):
    model.eval()
    model.cuda()
    x = torch.rand(input_size).cuda()
    x = model.maxpool(model.act1(model.bn1(model.conv1(x))))
    max_size = -1
    for blockidx in range(len(model.multiblocks)):
        for choiceidx in range(len(model.multiblocks[blockidx])):
            x = model.multiblocks[blockidx][choiceidx](x)
            max_size = max(max_size,np.prod(x.shape))
        x = model.multiblocks[blockidx][0](x)
    return max_size  * 4. / (1024 ** 2.)
    
def save_sharedfeature(model,model_type,loader,idxs,method,save_path=None):
    if 'resnet' in model_type:
        layers = [model.conv1, model.bn1, model.act1, model.maxpool]
    elif 'mobilenetv2' in model_type:
        layers = [model.conv_stem, model.bn1, model.act1, model.multiblocks[0][0]]
    former_layers = torch.nn.Sequential(*layers)
    former_layers = former_layers.cuda()
    features = []
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        if method == "AdaptiveNet":
            x = former_layers(input)
        else:
            x = input
        features.append((x,target))
    torch.save(features,save_path+str(idxs)+'.pth')

def load_data(model, model_type, data_dir, save_path, method, load_times=None, batch_size=None, data_len=500):
    t1 = time.time()
    model.eval()
    model.cuda()
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset_eval = torchvision.datasets.ImageFolder(root=data_dir,transform=data_transform)
    idxs = np.load('./npy/idxs.npy').tolist()[:data_len]
    if load_times is None:
        batch_size = get_batchsize(model,(2,3,224,224),0)
        load_times  = math.ceil(len(dataset_eval)/batch_size)
    load_img_num = math.ceil(data_len / load_times)
    for i in range(load_times-1):
        eval_set = Subset(dataset_eval, idxs[i*load_img_num:(i+1)*load_img_num])
        loader_eval = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False)
        save_sharedfeature(model,model_type,loader_eval,i, method = method, save_path=save_path)
    eval_set = Subset(dataset_eval, idxs[(load_times-1)*load_img_num:data_len])
    loader_eval = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    save_sharedfeature(model,model_type,loader_eval,load_times-1,method=method,save_path=save_path)
    # print('load_data end. time:{}'.format(time.time()-t1))


def subnet_latency(subnet,block_latency):
    lat = 0
    for blockidx in range(len(subnet)):
        if subnet[blockidx] != 99 and subnet[blockidx]!='9':
            lat += block_latency[blockidx][int(subnet[blockidx])]
    return lat

def subnet_path_to_int(subnet):
    block_len = len(subnet)
    re_subnet = []
    for i in range(block_len):
        if subnet[i] == 0:
            re_subnet.append(99)
        else:
            re_subnet.append(0)
    for i in range(block_len - 2):
        if re_subnet[i] != 99:
            if re_subnet[i+1] == 99 and re_subnet[i+2] == 99:
                re_subnet[i] = 2
            elif re_subnet[i+1] == 99:
                re_subnet[i] = 1
            else:
                re_subnet[i] = 0
        if re_subnet[-1] == 99:
            re_subnet[block_len-2] = 1
    return re_subnet

def subnet_int_to_str(subnet):
    s = ''
    for t in subnet:
        if t == 99:
            t = 9
        s += str(t)
    return s

def generate_subnet(path,block_len,block_latency,result,model_type='resnet50'):
    if len(path) == block_len:
        subnet = subnet_path_to_int(path)
        latency = subnet_latency(subnet,block_latency)
        result.append((subnet,latency))
    else:
        tmp_path = copy.deepcopy(path)
        tmp_path.append(1)
        generate_subnet(tmp_path,block_len,block_latency,result,model_type)
        if (len(path)>=2 and model_type == 'resnet50') or  (len(path)>=4 and model_type == 'mobilenetv2_100') :
            if path[-1]  + path[-2] != 0 :
                tmp_path = copy.deepcopy(path)
                tmp_path.append(0)
                generate_subnet(tmp_path,block_len,block_latency,result,model_type)
        elif (len(path)==1 and model_type == 'resnet50') or  (len(path)==3 and model_type == 'mobilenetv2_100'):                
            tmp_path = copy.deepcopy(path)
            tmp_path.append(0)
            generate_subnet(tmp_path,block_len,block_latency,result,model_type)
            
            
def get_subnets(sample_num=100,block_latency=None,model_type='resnet50',time_budget=0.9,low=0.9,up=1.1):      
    start = []
    result = []
    if model_type=='mobilenetv2_100':
        start.append(1)
        start.append(1)
    generate_subnet(start,len(block_latency),block_latency,result,model_type)
    result = sorted(result,key=lambda x: x[1])
    base_latency = result[-1][1]
    final_result = []
    for i in range(len(result)):
        if result[i][1] > low*time_budget*base_latency and result[i][1] < up*time_budget*base_latency:
            final_result.append(result[i][0])
    np.random.shuffle(final_result)
    # 10609 resnet50
    return final_result[:sample_num]

def get_subnet_pre(subnet,block_latency,pre_len=3):
    available_storage = 49 * 1024 
    data_size = 500
    batch_size = 4
    s = []
    for t in subnet:
        tmp = ''
        for k in t:
            if k == 99:
                k = 9
            tmp += str(k)
        s.append(tmp)
    subnet_int = subnet   
    subnet = s 
    subnet_len = len(subnet[0])
    re = []
    calculated_sub = set()             
    for i in range(len(subnet)):
        for j in range(1,subnet_len+1):
            cnt = 0
            if subnet[i][:j] in calculated_sub:
                continue
            for k in range(len(subnet)):
                if i != k:
                    if subnet[i][:j] == subnet[k][:j]:
                        cnt += 1
            if cnt != 0:
                time_cost = 0
                storage_coust = 0
                time_cost = subnet_latency(subnet[i][:j],block_latency)
                re.append((subnet[i][:j],cnt,cnt*time_cost,storage_coust,data_size/batch_size*cnt*0.005))
                calculated_sub.add(subnet[i][:j])
    re = sorted(re,key=lambda x: x[2])
    final_re = []
    cnt = 0 
    i = len(re) - 1
    count = 0
    while i >=0:
        if(len(re[i][0])>pre_len):
            final_re.append(re[i][0])
        i -= 1
    return final_re,subnet,subnet_int

def find_root(node, parent):
    if parent[node][4] == -1: 
        return node
    else:  
        return find_root(parent[node][4], parent)

def union(x, y, parent):
   
    x_root, y_root = find_root(x, parent), find_root(y, parent)
    if x_root == y_root:
        return 0
    else:
        parent[x_root][4] = y_root
        return 1

def get_subnet_tree(subnets,block_latency=None,pre_len=1):
    save_subnet_sub,subnets,subnet_int = get_subnet_pre(subnets,block_latency,pre_len=pre_len)
    tmp = []
    for (i,sn) in enumerate(subnets):
        tmp.append([sn,'-1',0,-1,-1,[],-1,i]) 
    subnets = tmp
    sub_len = len(subnets)
    for i in range(len(subnets)):
        for j in range(len(save_subnet_sub)):
            if save_subnet_sub[j] == subnets[i][0][:len(save_subnet_sub[j])] and len(save_subnet_sub[j]) > subnets[i][2]:
                subnets[i][1] = save_subnet_sub[j]
                subnets[i][2] = len(save_subnet_sub[j])
    for i in range(len(subnets)):
        for j in range(len(subnets)):
            if len(subnets[j][1]) <= len(subnets[i][1]) and i !=j:
                if subnets[j][1] == subnets[i][1][:len(subnets[j][1])] and len(subnets[j][1]) > subnets[i][3] and subnets[j][1] != '-1':
                    subnets[i][3] = len(subnets[j][1])
                    subnets[i][6] = j
    for i in range(sub_len):
        ans = union(i,subnets[i][6], subnets)
    for i in range(sub_len):
        if subnets[i][4] == -1:
            for j in range(sub_len):
                if i != j and len(subnets[j][1])<len(subnets[i][1]) and subnets[j][1] == subnets[i][1][:len(subnets[j][1])]:
                    subnets[i][4]=j
    for i in range(sub_len):
        for j in range(sub_len):
            if subnets[j][4] == i:
                subnets[i][5].append(j)
            
    root_subnet = []
    for i,sn in enumerate(subnets):
        if sn[4] == -1:
            root_subnet.append(i)
    for i in range(len(subnets)):
        subnets[i][0]=subnet_int[i]

    return subnets,root_subnet
        
