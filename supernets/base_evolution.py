import torch
import os
import pickle
import copy
import random
import ntpath
import numpy as np
import matplotlib.pyplot as plt
from configs import encode_config, decode_config
from tqdm import tqdm, trange 
from models.spade_model import SPADEModel
from metric import get_fid, get_coco_scores, get_cityscapes_mIoU
from utils import util


class BaseEvolution():
    def __init__(self, opt):
        # Evolution parameters
        self.evolution_sampling_flag = False
        self.population_size = self.opt.population_size 
        self.gen_num = self.opt.gen_num   
        self.p_crossover = self.opt.p_crossover
        self.p_mutation = self.opt.p_mutation
        self.eval_cnt = self.opt.eval_cnt 
        if 'horse2zebra' in self.opt.dataroot:
            self.eval_cnt = 2
        self.max_cache_size = 1000000

    '''
    Initialize population for EC
    '''
    def init_population(self, population_size=2):
        self.macs_cache = {}
        parents = []
        population, child_pool, macs_pool = [], [], []
        best_valids, best_infos = [], []

        for _ in trange(population_size, desc='Sample     '):
            sample = self.configs.sample() 
            while encode_config(sample) in self.child_pool_cache:  # self.child_pool_cache reinitialized at the begining of run_evolution
                sample = self.configs.sample()
            self.child_pool_cache.add(encode_config(sample))
            child_pool.append(sample)

        results = self.evaluate_population(child_pool)

        # Profile macs
        for i in range(population_size):
            sample = child_pool[i]
            macs = self.macs_cache.get(encode_config(sample))
            if macs is None:
                macs, _ = self.profile(sample, verbose=False)
                if len(self.macs_cache) < self.max_cache_size:
                    self.macs_cache[encode_config(sample)] = macs
            macs_pool.append(macs)

            population.append((results[i], child_pool[i], macs_pool[i]))
        
        return population


    def evaluate_population(self, child_pool):
        opt = self.opt
        results = []
        for child in tqdm(child_pool, position=1, desc='Evaluate child pool   ', leave=False):
            ret = self.evaluate_model_for_ec(child)
            results.append(ret)

        return results


    # Domination check
    def dominate(self, p, q):
        p_obj = [p[0]['obj'], p[2]] # obj(fid/mIoU), macs
        q_obj = [q[0]['obj'], q[2]] # obj(fid/mIoU), macs
        result = False
        for i, j in zip(p_obj, q_obj):
            if i < j:  # at least less in one dimension
                result = True
            elif i > j:  # not greater in any dimension, return false immediately
                return False
        return result


    def non_dominate_sorting(self, population):
        # find non-dominated sorted
        dominated_set = {}
        dominating_num = {}
        rank = {}
        for p in population:
            dominated_set[id(p)] = []
            dominating_num[id(p)] = 0

        sorted_pop = [[]]
        rank_init = 0
        for i, p in enumerate(population):
            for q in population[i + 1:]:
                if self.dominate(p, q):
                    dominated_set[id(p)].append(q)
                    dominating_num[id(q)] += 1
                elif self.dominate(q, p):
                    dominating_num[id(p)] += 1
                    dominated_set[id(q)].append(p)
            # rank 0
            if dominating_num[id(p)] == 0:
                rank[id(p)] = rank_init # rank set to 0
                sorted_pop[0].append(p)

        while len(sorted_pop[rank_init]) > 0:
            current_front = []
            for ppp in sorted_pop[rank_init]:
                for qqq in dominated_set[id(ppp)]:
                    dominating_num[id(qqq)] -= 1
                    if dominating_num[id(qqq)] == 0:
                        rank[id(qqq)] = rank_init + 1
                        current_front.append(qqq)
            rank_init += 1

            sorted_pop.append(current_front)

        return sorted_pop


    # Crowding distance
    def crowding_dist(self, objs):
        pop_size = len(objs[0])
        crowding_dis = np.zeros((pop_size, 1))

        obj_dim_size = len(objs)
        # crowding distance
        for m in range(obj_dim_size):
            obj_current = objs[m]
            sorted_idx = np.argsort(obj_current)  # sort current dim with ascending order
            obj_max = np.max(obj_current)
            obj_min = np.min(obj_current)

            # keep boundary point
            crowding_dis[sorted_idx[0]] = np.inf
            crowding_dis[sorted_idx[-1]] = np.inf
            for i in range(1, pop_size - 1):
                crowding_dis[sorted_idx[i]] = crowding_dis[sorted_idx[i]] + \
                                                        1.0 * (obj_current[sorted_idx[i + 1]] - \
                                                                obj_current[sorted_idx[i - 1]]) / (obj_max - obj_min)
        return crowding_dis


    # Environmental Selection
    def environmental_selection(self, population, n):     
        pop_sorted = self.non_dominate_sorting(population)
        selected = []
        for front in pop_sorted:
            if len(selected) < n:
                if len(selected) + len(front) <= n:
                    selected.extend(front)
                else:
                    # select individuals according to crowding distance
                    objs = []
                    objs.append([ind[0]['obj'] for ind in front]) # obj(fid/mIoU)
                    objs.append([ind[2] for ind in front]) # macs

                    crowding_dst = self.crowding_dist(objs)
                    k = n - len(selected)
                    dist_idx = np.argsort(crowding_dst, axis=0)[::-1]
                    for i in dist_idx[:k]:
                        selected.extend([front[i[0]]])
                    break
        return selected


    def variation(self, population, p_crossover, p_mutation):
        child_pool_set = set()

        offspring, child_pool, macs_pool = [], [], []
        len_pop = int(np.ceil(len(population) / 2) * 2) # deal with exception of not even number of individuals
        candidate_idx = np.random.permutation(len_pop)
        # ensure that the children are different from parents
        while len(child_pool_set)<len_pop:
            for i in range(int(len_pop/2)):
                # Crossover channels
                individual1 = population[candidate_idx[i]][1]
                individual2 = population[candidate_idx[-i-1]][1]
                child1, child2 = self.crossover_sample(individual1, individual2, p_crossover)

                # Mutation channels
                child1 = self.mutate_sample(child1, p_mutation)
                child2 = self.mutate_sample(child2, p_mutation)

                # Extract channels
                if encode_config(child1) not in self.child_pool_cache:
                    self.child_pool_cache.add(encode_config(child1))  # update child pool cache
                    child_pool_set.add(encode_config(child1))
                if encode_config(child2) not in self.child_pool_cache:
                    self.child_pool_cache.add(encode_config(child2)) # update child pool cache
                    child_pool_set.add(encode_config(child2))

        child_pool = [decode_config(child) for child in child_pool_set]
        child_pool = child_pool[:len(population)]

        # Evaluate child pool
        results = self.evaluate_population(child_pool)

        for i in range(len(child_pool)):
            # Update macs
            new_sample = child_pool[i]
            macs = self.macs_cache.get(encode_config(new_sample))
            if macs is None:
                macs, _ = self.profile(new_sample, verbose=False)
                if len(self.macs_cache) < self.max_cache_size:
                    self.macs_cache[encode_config(new_sample)] = macs
            macs_pool.append(macs) 

            offspring.append((results[i], child_pool[i], macs_pool[i]))

        return offspring
    

    '''
    augment the child pool via variation without evaluation
    '''
    def augmentation_variation(self, population, p_crossover, p_mutation):
        child_pool_set = set()
        # avoid generating repeated sample 
        for child in population:        
            child_pool_set.add(encode_config(child[1])) 

        offspring, child_pool, macs_pool = [], [], []
        len_pop = int(np.ceil(len(population) / 2) * 2) # deal with exception of not even number of individuals
        candidate_idx = np.random.permutation(len_pop)
        # ensure that the children are different from parents
        while len(child_pool_set)<len_pop:
            for i in range(int(len_pop/2)):
                # Crossover channels
                individual1 = population[candidate_idx[i]][1]
                individual2 = population[candidate_idx[-i-1]][1]
                child1, child2 = self.crossover_sample(individual1, individual2, p_crossover)

                # Mutation channels
                child1 = self.mutate_sample(child1, p_mutation)
                child2 = self.mutate_sample(child2, p_mutation)

                # Extract channels
                if encode_config(child1) not in child_pool_set:
                    child_pool_set.add(encode_config(child1))
                if encode_config(child2) not in child_pool_set:
                    child_pool_set.add(encode_config(child2))

        child_pool = [decode_config(child) for child in child_pool_set]
        child_pool = child_pool[:len(population)]

        for i in range(len(child_pool)):
            offspring.append((0, child_pool[i], 0))

        return offspring


    def crossover_sample(self, sample1, sample2, p_crossover):
        new_sample1 = copy.deepcopy(sample1)
        new_sample2 = copy.deepcopy(sample2)

        if np.random.random()<=p_crossover:
            for i in range(len(new_sample1['channels'])):
                choice = random.choice([0, 1])
                new_sample1['channels'][i] = sample2['channels'][i] if choice==1 else sample1['channels'][i]
                new_sample2['channels'][i] = sample1['channels'][i] if choice==1 else sample2['channels'][i]

        return new_sample1, new_sample2


    def mutate_sample(self, sample, mutate_prob):
        new_sample = copy.deepcopy(sample)
        for i in range(len(new_sample['channels'])):
            if random.random() < mutate_prob:
                new_sample['channels'][i] = self.configs.sample_layer(i)
            
        return new_sample


    def mmd_decision_making(self, pop):
        pop_objs = np.array([[x[0]['obj'], x[2]] for x in pop])
        N = pop_objs.shape[0]
        ideal = np.min(pop_objs, axis=0)
        nadir = np.max(pop_objs, axis=0)
        pop_objs = (pop_objs - ideal) / (nadir - ideal)
        mmd = np.sum(pop_objs, axis=1)
        knee_idx = np.argmin(mmd)
        arch_str = "_".join(str(x) for x in pop[knee_idx][1]["channels"])
        macs = pop[knee_idx][2]
        return arch_str, macs


    def save_population(self, population, epoch, gen):
        objs1 = [p[0]['obj'] for p in population] # obj(fid/mIoU)
        objs2 = [p[-1] for p in population] # macs
        # Save to file
        plt.figure()
        plt.plot(objs1, objs2, 'bo')
        plt.title("epoch:{}, generation:{}".format(epoch, gen))
        save_dir = os.path.join(self.opt.log_dir, 'evolution', str(epoch))
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'result_{}.png'.format(gen)))
        plt.close('all')

        pop_pkl = os.path.join(save_dir, 'population_{}.pkl'.format(gen))
        with open(pop_pkl, "wb") as f:
            pickle.dump(population, f)

        if epoch=="latest":
            arch_str, macs = self.mmd_decision_making(population)
            with open(os.path.join(save_dir, 'arch_str_{}.txt'.format(gen)), "w") as f:
                f.write(arch_str + "," + str(macs))


    def run_evolution(self, epoch): 
        self.child_pool_cache = set()  # this step resets child pool cache 
        
        population_history = []
        population = self.init_population(self.population_size)  # update child pool cache
        population_history.append(copy.deepcopy(population))

        for gen in range(self.gen_num):
            # Variation (evluation for the offspring included)
            offspring = self.variation(population, self.p_crossover, self.p_mutation)  # will update child pool cache

            # P+Q
            population.extend(offspring)

            # Environmental Selection
            population = self.environmental_selection(population, self.population_size)
            population_history.append(copy.deepcopy(population))

            # Save population
            self.save_population(population, epoch, gen)
            self.save_population(population, "latest", "latest")

        evolved_pool = [sample[1] for sample in population] # extract config
        return evolved_pool, population_history
                    
    def evolution(self, epoch):
        if self.opt.config_str is None:
            if epoch>=self.opt.warmup_epochs:
                # sampling
                if hasattr(self, "evolved_pool") and self.evolution_sampling_flag == True:
                    offspring_unevaluated = self.augmentation_variation(self.population_history[-1], self.p_crossover, self.p_mutation)
                    augmented_pool_unevaluated = [offspring[1] for offspring in offspring_unevaluated]
                    self.evolved_pool = self.evolved_pool_evaluated + augmented_pool_unevaluated  # combine two lists
                        
                # start evolution and save population history
                if epoch % self.opt.evolution_epoch_freq == 0:
                    self.evolved_pool, self.population_history = self.run_evolution(epoch) # reset sample cache
                    self.evolved_pool_evaluated = copy.deepcopy(self.evolved_pool)
                    self.evolution_sampling_flag = True

