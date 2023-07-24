import torch
import time,sys
from tqdm import tqdm
import numpy as np
import tools.mytools
from mytimm.utils import *
import random
import copy
from tools import mytools
from tools import global_var
import sys


class EvolutionFinder:
    def __init__(self, mutate_prob=0.1, population_size=100, parent_ratio=0.25, mutation_ratio=0.5, searching_times=50,
                 max_trying_times=1000, time_budget=0.01, batch_size=4, branch_choices=None, lats=None, model_lens=None):  # parent_ratio*2+mutation_ratio=1.0
        self.mutate_prob = mutate_prob
        self.population_size = population_size
        self.parent_ratio = parent_ratio
        self.mutation_ratio = mutation_ratio
        self.searching_times = searching_times
        self.max_trying_times = max_trying_times
        self.time_budget = time_budget
        self.test_input = torch.randn(batch_size, 3, 224, 224)
        self.branch_choices = branch_choices  # [[0, 1, 2], [0, 1]...[0, 1]], len=len(subnetchoices)
        self.population = []
        self.latencys = []
        self.accs = []
        self.lats = lats
        self.model_lens = model_lens
        self.prune_points = None if self.model_lens is None else self.get_prune_points(model_lens)
        self.finished_subnets = set()

    def get_prune_points(self, model_lens):  # model_lens:[1, 2, 2, 3, 3, 3, 2, 2, 1...]
        old_model_len = model_lens[0]
        prune_points = []
        for blockidx in range(len(model_lens)):
            if model_lens[blockidx] != old_model_len:
                old_model_len = model_lens[blockidx]
                prune_points.append(blockidx)
        return prune_points

    def get_subnet_latency(self, originalmodel, subnet):
        latency = self.lats[0]
        for blockidx in range(len(subnet)):
            if subnet[blockidx] != 99:
                latency += self.lats[blockidx+1][subnet[blockidx]]
        latency += self.lats[-1]
        return latency

    def init_population(self, args,model):
        if args.model == 'resnet50':
            lat = self.lats[1:-1]
        elif args.model == 'mobilenetv2_100':
            lat = self.lats[1:-1]
        self.population = mytools.get_subnets(sample_num=self.population_size,block_latency=lat,model_type=args.model) 
    def init_population_baseline(self, args,model):
        example_subnet = model.generate_random_subnet()
        self.population = mytools.generate_subnets(sample_num=self.population_size, model_len=len(example_subnet))

    def mutate_sample(self, model, oldsample, get_latency=True,model_type='resnet50'):  # [0..len(subnet)]
            sample = copy.deepcopy(oldsample)
            for _ in range(self.max_trying_times):
                blockidx = 0  # 1 for mbv2 and 0 for resnets
                if model_type == 'mobilenetv2_100':
                    blockidx = 1
                while blockidx < len(sample):
                    if (sample[blockidx] != 99) and (random.random() < self.mutate_prob):
                        original_choice = sample[blockidx]
                        bolck_time = -1*self.lats[blockidx+1][original_choice]
                        sample[blockidx] = random.choice(self.branch_choices[blockidx])
                        bolck_time += self.lats[blockidx+1][sample[blockidx]]
                        if sample[blockidx] == 2:
                            if sample[blockidx + 1] != 99:
                                bolck_time -= self.lats[blockidx+1+1][sample[blockidx+1]]
                            if sample[blockidx + 2] != 99:
                                bolck_time -= self.lats[blockidx+1+2][sample[blockidx+2]]
                            sample[blockidx + 1] = 99
                            sample[blockidx + 2] = 99
                            blockidx += 3
                        elif sample[blockidx] == 1:
                            if sample[blockidx + 1] != 99:
                                bolck_time -= self.lats[blockidx+1+1][sample[blockidx+1]]
                            sample[blockidx + 1] = 99
                            blockidx += 2
                        elif sample[blockidx] == 0:
                            blockidx += 1
                        else:  # [-2, -1, 0], pruning cases
                            for i in range(1, self.model_lens[blockidx]):
                                sample[blockidx + i] = 99
                            blockidx += self.model_lens[blockidx]
                        while blockidx <= len(sample) - 1:
                            if sample[blockidx] == 99:
                                sample[blockidx] = 0
                                bolck_time += self.lats[blockidx+1][sample[blockidx]]
                                blockidx += 1 
                            else:
                                break
                        tmp = []
                        if abs(bolck_time) < 1e-05:
                            continue    
                        if bolck_time > 0:
                            for i in range(2,len(sample)):
                                if sample[i] == 0:
                                    if sample[i-1] == 0:
                                        tmp.append(i)
                                    elif sample[i-1] == 9 and sample[i-2] == 1:
                                        tmp.append(i)
                            if len(tmp) >0:
                                for idx in range(len(tmp)):
                                    re_idx = -1
                                    re_min = 100
                                    if abs(bolck_time-self.lats[tmp[idx]+1][0]) < re_min:
                                        re_min = abs(bolck_time-self.lats[tmp[idx]+1][0])
                                        re_idx = tmp[idx]
                                if sample[re_idx-1] == 0:
                                    sample[re_idx-1] = 1
                                    sample[re_idx] = 99
                                else:
                                    sample[re_idx-2] = 2
                                    sample[re_idx] = 99
                        else:
                            for i in range(2,len(sample)):
                                if sample[i] == 9:
                                    if sample[i-1] == 1:
                                        tmp.append(i)
                                    elif sample[i-1] == 9 and sample[i-2] == 2:
                                        tmp.appen(i)
                            if len(tmp) >0:
                                for idx in range(len(tmp)):
                                    re_idx = -1
                                    re_min = 100
                                    if abs(bolck_time+self.lats[tmp[idx]+1][0]) < re_min:
                                        re_min = abs(bolck_time+self.lats[tmp[idx]+1][0])
                                        re_idx = tmp[idx]
                                if sample[re_idx-1] == 1:
                                    sample[re_idx-1] = 0
                                    sample[re_idx] = 0
                                else:
                                    sample[re_idx-2] = 1
                                    sample[re_idx] = 0                      
                    else:
                        blockidx += 1
                if get_latency:
                    latency = self.get_subnet_latency(model, sample)
                    if latency < self.time_budget:
                        return sample, latency
                else:
                    return sample
    def mutate_sample_baseline(self, model, sample, get_latency=True,model_type='resnet50'):  # [0..len(subnet)]
        for _ in range(self.max_trying_times):
            blockidx = 0
            if model_type == 'mobilenetv2_100':
                    blockidx = 1
            while blockidx < len(sample):
                if (sample[blockidx] != 99) and (random.random() < self.mutate_prob):
                    sample[blockidx] = random.choice(self.branch_choices[blockidx])
                    if sample[blockidx] == 2:
                        sample[blockidx + 1] = 99
                        sample[blockidx + 2] = 99
                        blockidx += 3

                    elif sample[blockidx] == 1:
                        sample[blockidx + 1] = 99
                        blockidx += 2
                    else:
                        blockidx += 1
                    if blockidx <= len(sample) - 1:
                        if sample[blockidx] == 99:
                            sample[blockidx] = 0
                            blockidx += 1
                            if blockidx <= len(sample) - 1:
                                if sample[blockidx] == 99:
                                    sample[blockidx] = 0
                                    blockidx += 1
                else:
                    blockidx += 1
            if get_latency:
                latency = self.get_subnet_latency(model, sample)
                if latency < self.time_budget:
                    return sample, latency
            else:
                return sample
    def cross_over(self, model, sample1, sample2, get_latency=True):
        for _ in range(self.max_trying_times):
            blockidx = 0
            new_sample = []
            while blockidx < len(sample1):
                if sample1[blockidx] == 99:
                    block_choice = sample2[blockidx]
                elif sample2[blockidx] == 99:
                    block_choice = sample1[blockidx]
                else:
                    block_choice = random.choice([sample1[blockidx], sample2[blockidx]])
                new_sample.append(block_choice)
                if block_choice == 1:
                    new_sample.append(99)
                    blockidx += 2
                elif block_choice == 2:
                    new_sample += [99, 99]
                    blockidx += 3
                else:
                    blockidx += 1
            
            if get_latency:
                latency = self.get_subnet_latency(model, new_sample)
                if latency < self.time_budget:
                    return new_sample, latency
            else:
                return new_sample

    def dfs(self,root,subnets,model, validate, data_loader, args, loss_fn,accs,final_re,subnettree_cur_depth):
        inter_features = global_var.get_value('inter_features')
        global total_time 
        if len(subnets[root][5]) == 0:
            return
        while(len(subnets[root][5])>=1):
            idx = subnets[root][5].pop()
            global_var.set_value('subnet', subnets[idx][0])
            global_var.set_value('subnet_pre', subnets[idx][1])
            if subnets[idx][4] != -1:
                global_var.set_value('subnet_shared_block', subnets[subnets[idx][4]][1])
            else:
                global_var.set_value('subnet_shared_block', '-1')
            sum,count, latency=validate(model, data_loader, subnets[idx][0], args, loss_fn,self.lats)
            accs[subnets[idx][-1]][0] +=sum
            accs[subnets[idx][-1]][1] +=count
            final_re[subnets[idx][-1]][-1] = latency
            final_re[subnets[idx][-1]][1] = subnets[idx][0]
            self.dfs(idx,subnets,model, validate, data_loader, args, loss_fn,accs,final_re,subnettree_cur_depth+1)
            if  subnets[idx][1] != subnets[subnets[idx][4]][1]:
                del inter_features[subnets[idx][1]]

    def val_subnets(self,subnets,model, validate, args, loss_fn):
        if args.model == 'resnet50':
            lat = self.lats[1:-1]
        elif args.model == 'mobilenetv2_100':
            lat = self.lats[1:-1]
        subnet_tree,root_subnet = mytools.get_subnet_tree(subnets,lat,pre_len=args.pre_len)
        #[sn,'-1',0,-1,-1,[],-1,i]
        final_re = [[0,0,0,0] for i in range(len(subnets))]
        accs = [[0,0] for i in range(len(subnets))]
        for i in range(args.load_times):
            data_loader = torch.load(args.save_path+str(i)+'.pth')  
            inter_features=dict()   
            global_var.set_value('inter_features', inter_features)
            subnets = copy.deepcopy(subnet_tree)
            for root in root_subnet:
                global_var.set_value('subnet', subnets[root][0])
                global_var.set_value('subnet_pre', subnets[root][1])
                if subnets[root][4] != -1:
                    global_var.set_value('subnet_shared_block', subnets[subnets[root][4]][1])
                else:
                    global_var.set_value('subnet_shared_block', '-1')
                sum,count, latency=validate(model, data_loader, subnets[root][0], args, loss_fn,self.lats)
                accs[subnets[root][-1]][0] +=sum
                accs[subnets[root][-1]][1] +=count
                final_re[subnets[root][-1]][-1] = latency
                final_re[subnets[root][-1]][1] = subnets[root][0]
                self.dfs(root,subnets,model, validate, data_loader, args, loss_fn,accs,final_re,1)
                del inter_features[subnets[root][1]]
        for i in range(len(subnets)):
            final_re[i][2] = accs[i][0]/accs[i][1]
            score = (final_re[i][2] / 100. - final_re[i][-1] / self.time_budget) if final_re[i][-1] > self.time_budget else final_re[i][2]
            final_re[i][0] = score
        return final_re
        
    def val_subnets_baseline(self,subnets,model, validate, args, loss_fn):
        start_time = time.time()
        final_re = [[0,0,0,0] for i in range(len(subnets))]
        accs = [[0,0] for i in range(len(subnets))]
        for i in range(args.baseline_load_times):
            data_loader = torch.load(args.baseline_save_path+str(i)+'.pth')  
            for i in range(len(subnets)):
                sum,count, latency=validate(model, data_loader, subnets[i], args, loss_fn,self.lats,infer_type=0)
                accs[i][0] +=sum
                accs[i][1] +=count
                final_re[i][-1] = latency
                final_re[i][1] = subnets[i]
        for i in range(len(subnets)):
            final_re[i][2] = accs[i][0]/accs[i][1]
            score = (final_re[i][2] / 100. - final_re[i][-1] / self.time_budget) if final_re[i][-1] > self.time_budget else final_re[i][2]
            final_re[i][0] = score
        return final_re
            
    def evolution_search(self, model, validate, args, loss_fn):  # validate is the function to get the acc and latency of a subnet
        self.init_population(args,model)
        start_time = time.time()
        mutation_number = int(round(self.population_size * self.mutation_ratio))
        parent_size = int(round(self.population_size * self.parent_ratio))
        for subnet in self.population:
            str_sub = mytools.subnet_int_to_str(subnet)
            self.finished_subnets.add(str_sub)
        best_valids = [-100]
        best_info = None
        times = [] 
        t1 = time.time()
        self.population = self.val_subnets(self.population,model,validate,args,loss_fn)
        times.append((time.time()-t1,len(self.population)))
        for iter in range(self.searching_times):
            parents = sorted(self.population, key=lambda x: x[0])[::-1][:parent_size]  # reverted sort
            acc = parents[0][2]
            latency = parents[0][3]
            if acc > best_valids[-1] and parents[0][0] > 0:
                best_valids.append(acc)
                best_info = parents[0]
            if iter > 0:
                print('[{},{},{}],'.format(iter,best_info,time.time()-start_time))
            self.population = parents
            subnets = []
            for _ in range(mutation_number):
                subnet = self.population[np.random.randint(parent_size)][1]
                ans = 0
                while ans < self.max_trying_times:
                    child_subnet = self.mutate_sample(model, subnet, get_latency=False,model_type=args.model)
                    str_sam = mytools.subnet_int_to_str(child_subnet)
                    if str_sam not in self.finished_subnets:
                        break
                    ans += 1
                self.finished_subnets.add(str_sam)
                subnets.append(child_subnet)

            for _ in range(self.population_size - mutation_number):
                ans = 0
                while ans < self.max_trying_times:
                    father_subnet = self.population[np.random.randint(parent_size)][1]
                    mother_subnet = self.population[np.random.randint(parent_size)][1]
                    child_subnet = self.cross_over(model, father_subnet, mother_subnet, get_latency=False)
                    str_sam = mytools.subnet_int_to_str(child_subnet)
                    if str_sam not in self.finished_subnets:
                        break
                    ans += 1
                self.finished_subnets.add(str_sam)
                subnets.append(child_subnet)
            t1 = time.time()
            re = self.val_subnets(subnets,model,validate,args,loss_fn)
            times.append((time.time()-t1,len(subnets)))
            for _ in re:
                self.population.append(_)
        parents = sorted(self.population, key=lambda x: x[0])[::-1][:parent_size]  # reverted sort
        acc = parents[0][2]
        latency = parents[0][3]
        best_valids.append(acc)
        best_info = parents[0]
        print('[{},{},{}],'.format(self.searching_times,best_info,time.time()-start_time))
        return best_valids, best_info
        
    def evolution_search_baseline0(self, model, validate, data_loader, args, loss_fn):  
        self.init_population_baseline(args,model)
        mutation_number = int(round(self.population_size * self.mutation_ratio))
        parent_size = int(round(self.population_size * self.parent_ratio))

        best_valids = [-100]
        best_info = None
        for subnetidx in range(len(self.population)):
            acc, latency = validate(model, subnet=self.population[subnetidx], loader=data_loader, args=args, loss_fn=loss_fn, lats=self.lats) 
            # score = acc - latency / self.time_budget
            score = (acc / 100. - latency / self.time_budget) if latency > self.time_budget else acc
            self.population[subnetidx] = (score, self.population[subnetidx], latency, acc)

        for iter in tqdm(range(self.searching_times), desc='searching for time budget %s' % (self.time_budget)):
            parents = sorted(self.population, key=lambda x: x[0])[::-1][:parent_size]  # reverted sort
            acc = parents[0][3]
            latency = parents[0][2]
            print('iter: {} Acc: {} Latency/TimeBudget: {}'.format(iter, acc, latency / self.time_budget))

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]

            self.population = parents
            for _ in range(mutation_number):
                subnet = self.population[np.random.randint(parent_size)][1]
                child_subnet = self.mutate_sample_baseline(model, subnet, get_latency=False,model_type=args.model)
                acc, latency = validate(model,data_loader,child_subnet,args=args, loss_fn=loss_fn, lats=self.lats)
                score = (acc / 100. - latency / self.time_budget) if latency > self.time_budget else acc
                self.population.append((score, child_subnet, latency, acc))

            for _ in range(self.population_size - mutation_number):
                father_subnet = self.population[np.random.randint(parent_size)][1]
                mother_subnet = self.population[np.random.randint(parent_size)][1]
                child_subnet = self.cross_over(model, father_subnet, mother_subnet, get_latency=False)
                acc, latency = validate(model,data_loader,child_subnet,args=args, loss_fn=loss_fn, lats=self.lats)
                score = (acc / 100. - latency / self.time_budget) if latency > self.time_budget else acc
                self.population.append((score, child_subnet, latency, acc))
        return best_valids, best_info

    def evolution_search_baseline1(self, model, validate, args, loss_fn):  
        start_time = time.time()
        self.init_population_baseline(args,model)
        mutation_number = int(round(self.population_size * self.mutation_ratio))
        parent_size = int(round(self.population_size * self.parent_ratio))
        for subnet in self.population:
            str_sub = mytools.subnet_int_to_str(subnet)
            self.finished_subnets.add(str_sub)
        best_valids = [-100]
        best_info = None
        self.population = self.val_subnets_baseline(self.population,model,validate,args,loss_fn)

        for iter in tqdm(range(self.searching_times), desc='searching for time budget %s' % (self.time_budget)):
            parents = sorted(self.population, key=lambda x: x[0])[::-1][:parent_size]  # reverted sort
            acc = parents[0][3]
            latency = parents[0][2]
            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]
            print('[{},{},{}],'.format(iter,best_info,time.time()-start_time))

            self.population = parents
            subnets = []
            for _ in range(mutation_number):
                subnet = self.population[np.random.randint(parent_size)][1]
                ans = 0
                while ans < self.max_trying_times:
                    child_subnet = self.mutate_sample_baseline(model, subnet, get_latency=False,model_type=args.model)
                    str_sam = mytools.subnet_int_to_str(child_subnet)
                    if str_sam not in self.finished_subnets:
                        break
                    ans += 1
                self.finished_subnets.add(str_sam)
                subnets.append(child_subnet)

            for _ in range(self.population_size - mutation_number):
                ans = 0
                while ans < self.max_trying_times:
                    father_subnet = self.population[np.random.randint(parent_size)][1]
                    mother_subnet = self.population[np.random.randint(parent_size)][1]
                    child_subnet = self.cross_over(model, father_subnet, mother_subnet, get_latency=False)
                    str_sam = mytools.subnet_int_to_str(child_subnet)
                    if str_sam not in self.finished_subnets:
                        break
                    ans += 1
                self.finished_subnets.add(str_sam)
                subnets.append(child_subnet)
            re = self.val_subnets_baseline(subnets,model,validate,args,loss_fn)
            for _ in re:
                self.population.append(_)
        parents = sorted(self.population, key=lambda x: x[0])[::-1][:parent_size]  # reverted sort
        acc = parents[0][2]
        latency = parents[0][3]
        best_valids.append(acc)
        best_info = parents[0]
        print('[{},{},{}],'.format(self.searching_times,best_info,time.time()-start_time))
        return best_valids, best_info
        