import numpy as np
import math
import random
import torch
import copy
import time
from tools import mytools
from tools import global_var

class SA:
    def __init__(self,population_size = 100, tmp = 500,tmp_min = 0.5, alpha = 0.5, perturbation_prob=0.1,time_budget=0.01, batch_size=4, branch_choices=None,lats=None): 
        self.tmp = tmp
        self.tmp_min = tmp_min
        self.alpha = alpha 
        self.branch_choices = branch_choices
        self.time_budget = time_budget
        self.perturbation_prob = perturbation_prob
        self.mutate_prob = perturbation_prob
        self.population_size = population_size

        self.lats = lats
        self.max_trying_times = 1000
        self.finished_subnets = set()
    def perturbationbaseline(self,sample,model_type='resnet50'):
        blockidx = 0 
        if model_type == 'mobilenetv2_100':
                blockidx = 1
        while blockidx < len(sample):
            if (sample[blockidx] != 99) and (random.random() < self.perturbation_prob):
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
        return sample
    def perturbation(self,sample,model_type='resnet50'):
        # if random.random() < self.mutate_prob:
        sample = copy.deepcopy(sample)
        for _ in range(self.max_trying_times):
            # for blockidx in range(len(sample)):
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
                            #print('blocktime:{} abs:{} lats:{}'.format(bolck_time,re_min,self.lats[re_idx+1][0]))
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
                            # print('blocktime:{} abs:{} lats:{}'.format(bolck_time,re_min,self.lats[re_idx+1][0])) 
                else:
                    blockidx += 1
        return sample

    def judge(self,deltaE,T):
        if deltaE < 0:
            return 1
        else: 
            probability = math.exp(-deltaE/T) #以自然常数e为底的指数函数
            if probability > random.random(): #当T变小时，probability在减小，于是接受更差的解的概率值越小，退火过程趋于稳定
                return 1
            else:
                return 0
    def cal_energy_baseline(self,model, validate,data_loader,subnet, args, loss_fn):
        acc = []
        latency = []
        score = []
        total_time = []
        for sn in subnet:
            sum,count, latency_=validate(model, data_loader, sn, args, loss_fn,self.lats,infer_type=0)
            acc_ = sum / count
            score_ = (acc_ / 100. - latency_ / self.time_budget) if latency_ > self.time_budget else acc_
            score.append(score_)
            latency.append(latency_)
            acc.append(acc_)
            total_time.append(time.time()-self.start_time)
        return score,latency,acc,total_time
    def dfs(self,root,subnets,model, validate, data_loader, args, loss_fn,accs,final_re):
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
            #print(accs[subnets[root][-1]][1],count)
            accs[subnets[idx][-1]][0] +=sum
            accs[subnets[idx][-1]][1] +=count
            final_re[subnets[idx][-1]][-1] = latency
            final_re[subnets[idx][-1]][1] = subnets[idx][0]
            self.dfs(idx,subnets,model, validate, data_loader, args, loss_fn,accs,final_re)
            if  subnets[idx][1] != subnets[subnets[idx][4]][1]:
                del inter_features[subnets[idx][1]]
    def cal_energy(self,model, validate,subnets, args, loss_fn):
        if args.model == 'resnet50':
            lat = self.lats[1:-1]
        elif args.model == 'mobilenetv2_100':
            lat = self.lats[1:-1]
        subnet_tree,root_subnet = mytools.get_subnet_tree(subnets,lat,pre_len=args.pre_len)
        #[sn,'-1',0,-1,-1,[],-1,i]
        final_re = [[0,0,0,0] for i in range(len(subnets))]
        accs = [[0,0] for i in range(len(subnets))]
        acc = []
        latency = []
        scores = []
        for i in range(args.load_times):
            data_loader = torch.load(args.save_path+str(i)+'.pth')  
            print(args.save_path+str(i)+'.pth')
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
                sum,count, latency=validate(model, data_loader, subnets[root][0], args, loss_fn, self.lats)
                accs[subnets[root][-1]][0] +=sum
                accs[subnets[root][-1]][1] +=count
                final_re[subnets[root][-1]][-1] = latency
                final_re[subnets[root][-1]][1] = subnets[root][0]
                self.dfs(root,subnets,model, validate, data_loader, args, loss_fn,accs,final_re)
                del inter_features[subnets[root][1]]
        for i in range(len(subnets)):
            final_re[i][2] = accs[i][0]/accs[i][1]
            score = (final_re[i][2] / 100. - final_re[i][-1] / self.time_budget) if final_re[i][-1] > self.time_budget else final_re[i][2]
            acc.append(accs[i][0]/accs[i][1])
            scores.append(score)
        return scores,acc
    def optimization_baseline(self,model, validate, args, loss_fn):
        counter = 0
        max_score = -1
        max_score_time = -1
        data_loader = torch.load('./npy/baseline-batch-0.pth')  
        
        ans = 0
        self.start_time = time.time()
        example_subnet = model.generate_random_subnet()
        sample_subnets = mytools.generate_subnets(sample_num=self.population_size, model_len=len(example_subnet))
        for subnet in sample_subnets:
            ans += 1
            str_sub = mytools.subnet_int_to_str(subnet)
        scores,latencys,accs,total_time = self.cal_energy_baseline(model, validate,data_loader, sample_subnets, args, loss_fn)
        for i in range(len(scores)):
             if scores[i] > max_score:
                max_score = scores[i]
                max_score_time = total_time[i]
        results = []
        old_scores = scores
        old_subnets = sample_subnets
        k = 5
        while(self.tmp >= self.tmp_min):    
            for inx in range(k):
                new_subnets = []
                for old_subnet in old_subnets:
                    ans = 0
                    while ans < self.max_trying_times:
                        new_subnet = self.perturbation(copy.deepcopy(old_subnet))
                        str_sam = mytools.subnet_int_to_str(new_subnet)
                        if str_sam not in self.finished_subnets:
                            break
                        ans += 1
                    ans += 1
                    self.finished_subnets.add(str_sam)
                    new_subnets.append(new_subnet)
                new_scores,latencys,accs,total_time = self.cal_energy_baseline(model, validate, data_loader, new_subnets, args, loss_fn)
                deltaE = []
                for i in range(len(old_scores)):
                    deltaE.append(old_scores[i]-new_scores[i])
                    if new_scores[i] > max_score:
                        max_score = new_scores[i]
                        max_score_time = total_time[i]
                for i in range(len(deltaE)):
                    if self.judge(deltaE[i],self.tmp) == 1:
                        old_subnets[i] = new_subnets[i]
                        old_scores[i] = new_scores[i]
            counter += 1
            self.tmp = self.tmp * self.alpha
            print('[{},{},{}],'.format(counter,max_score,time.time()-self.start_time))
    def init_population(self, args,model):
        if args.model == 'resnet50':
            lat = self.lats[1:-1]
        elif args.model == 'mobilenetv2_100':
            lat = self.lats[1:-1]
        return mytools.get_subnets(sample_num=self.population_size,block_latency=lat,model_type=args.model) 
    
    def optimization(self,model, validate, args, loss_fn):
        counter = 0
        max_score = -1
        best_subnet = []
        max_score_time = -1
        ans = 0
        self.start_time = time.time()
        sample_subnets = self.init_population(args,model)
        for subnet in sample_subnets:
            ans += 1
            str_sub = mytools.subnet_int_to_str(subnet)
            self.finished_subnets.add(str_sub)
        scores,accs = self.cal_energy(model, validate,sample_subnets, args, loss_fn)
        for i in range(len(scores)):
             if scores[i] > max_score:
                max_score = scores[i]
                max_score_time = time.time()-self.start_time
        results = []
        old_scores = scores
        old_subnets = sample_subnets
        k = 1
        while(self.tmp >= self.tmp_min):    
            for inx in range(k):
                new_subnets = []
                for old_subnet in old_subnets:
                    ans = 0
                    while ans < self.max_trying_times:
                        new_subnet = self.perturbation(copy.deepcopy(old_subnet))
                        str_sam = mytools.subnet_int_to_str(new_subnet)
                        if str_sam not in self.finished_subnets:
                            break
                        ans += 1
                    ans += 1
                    self.finished_subnets.add(str_sam)
                    new_subnets.append(new_subnet)
                new_scores,accs = self.cal_energy(model, validate, new_subnets, args, loss_fn)
                deltaE = []
                for i in range(len(old_scores)):
                    deltaE.append(old_scores[i]-new_scores[i])
                    if new_scores[i] > max_score:
                        max_score = new_scores[i]
                        best_subnet = new_subnets[i]
                        max_score_time = time.time()-self.start_time
                for i in range(len(deltaE)):
                    if self.judge(deltaE[i],self.tmp) == 1:
                        old_subnets[i] = new_subnets[i]
                        old_scores[i] = new_scores[i]
            counter += 1
            self.tmp = self.tmp * self.alpha
            print('[{},{},{},{}]'.format(counter,max_score,best_subnet,time.time()-self.start_time))
        return max_score,best_subnet