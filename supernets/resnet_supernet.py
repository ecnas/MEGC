import ntpath
import os
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import gather, parallel_apply, replicate
from tqdm import tqdm

from configs import decode_config
from configs.resnet_configs import get_configs
from configs.single_configs import SingleConfigs
from distillers.base_resnet_distiller import BaseResnetDistiller
from metric import get_fid, get_cityscapes_mIoU
from models.modules.super_modules import SuperConv2d
from utils import util

######################  Test Subnet ######################
# def test_fid(restore_student_G_path=opt.restore_student_G_path, config_str, input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.student_ngf, inception_model=model.inception_model, device=model.device):
def test_fid(config_str, student_G, input_nc, output_nc, ngf, inception_model, device):
    # Construct submodel
    import copy
    import numpy as np
    from data import create_dataloader
    from configs import decode_config
    from models.modules.resnet_architecture.mobile_resnet_generator import MobileResnetGenerator
    from models.modules.resnet_architecture.sub_mobile_resnet_generator import SubMobileResnetGenerator
    # opt.config_str = config_str  # "64_32_32_48_48_64_48_24"
    config = decode_config(config_str)
    # input_nc, output_nc = opt.input_nc, opt.output_nc
    # super_model = MobileResnetGenerator(input_nc, output_nc, ngf=ngf, norm_layer=torch.nn.InstanceNorm2d, n_blocks=9)
    super_model = copy.deepcopy(student_G)
    super_model.eval()
    sub_model = SubMobileResnetGenerator(input_nc, output_nc, config=config, norm_layer=torch.nn.InstanceNorm2d, n_blocks=9)
    
    from utils.util import load_network
    from export import transfer_weight

    # load_network(super_model, restore_student_G_path)
    transfer_weight(super_model, sub_model)
    # model.netG_student = sub_model

    # import pdb; pdb.set_trace()
    # Test submodel
    from options.test_options import TestOptions
    from models import create_model
    from metric import create_metric_models, get_cityscapes_mIoU, get_coco_scores, get_fid
    import ntpath
    class Options:
        pass
    
    optTest = Options()
    optTest.model = 'test'
    optTest.netG = 'sub_mobile_resnet_9blocks'
    optTest.ngf = 64
    optTest.dataroot = "database/horse2zebra/valA"
    optTest.dataset_mode = "single"
    optTest.config_str = config_str
    optTest.real_stat_path = "real_stat/horse2zebra_B.npz"
    optTest.restore_G_path = None # opt.restore_student_G_path
    optTest.phase = 'val'
    optTest.direction = 'AtoB'
    optTest.input_nc = 3
    optTest.preprocess = 'resize_and_crop'
    optTest.load_size = 256
    optTest.crop_size = 256
    optTest.no_flip = True
    optTest.batch_size = 1
    optTest.serial_batches = True
    optTest.num_threads = 4
    optTest.max_dataset_size = -1

    optTest.gpu_ids = [0]
    optTest.isTrain = False
    optTest.output_nc = 3
    optTest.norm = 'instance'
    optTest.dropout_rate = 0
    optTest.need_profile = True

    dataloaderTest = create_dataloader(optTest)
    modelTest = create_model(optTest)
    modelTest.netG = sub_model
    modelTest.netG.to(device)
    modelTest.netG.eval()
    modelTest.setup(optTest, verbose=False)

    # import pdb; pdb.set_trace()
    config = decode_config(optTest.config_str)
    fakes, names = [], []
    for i, data in enumerate(tqdm(dataloaderTest)):
        modelTest.set_input(data)  # unpack data from data loader
        if i == 0 and optTest.need_profile:
            macs, params = modelTest.profile(config)
        modelTest.test(config)  # run inference
        visuals = modelTest.get_current_visuals()  # get image results
        generated = visuals['fake_B'].cpu()
        fakes.append(generated)
        # if i==2:
        #     break
    npz = np.load(optTest.real_stat_path)
    fid = get_fid(fakes, inception_model, npz, device, optTest.batch_size, fast=False)
    print('fid score: %.2f' % fid, flush=True)
    # import pdb; pdb.set_trace()
    return fid, macs, params
######################  Test Subnet ######################



class ResnetSupernet(BaseResnetDistiller):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(ResnetSupernet, ResnetSupernet).modify_commandline_options(parser, is_train)
        parser.set_defaults(norm='instance', student_netG='super_mobile_resnet_9blocks',
                            dataset_mode='aligned', log_dir='logs/supernet')
        return parser

    def __init__(self, opt):
        assert 'super' in opt.student_netG
        super(ResnetSupernet, self).__init__(opt)
        self.best_fid_largest = 1e9
        self.best_fid_smallest = 1e9
        self.best_mIoU_largest = -1e9
        self.best_mIoU_smallest = -1e9
        self.fids_largest, self.fids_smallest = [], []
        self.mIoUs_largest, self.mIoUs_smallest = [], []

        if opt.config_set is not None:
            assert opt.config_str is None
            self.configs = get_configs(opt.config_set)
            self.opt.eval_mode = 'both'
        else:
            assert opt.config_str is not None
            self.configs = SingleConfigs(decode_config(opt.config_str))
            self.opt.eval_mode = 'largest'

        # Evolution Parameters
        self.evolution_sampling_flag = False # default False for evolutionary sampling
        self.population_size = self.opt.population_size # 100
        self.gen_num = 30   # 1 # Number of generations for evolution during supernet training
        self.p_crossover = 1
        self.p_mutation = 0.5
        self.eval_cnt = 1  # population evaluation on only one sample
        if 'horse2zebra' in self.opt.dataroot:
            self.eval_cnt = 2

        self.cal_loss = self.opt.eval_loss

    '''
    Initialize population for EC
    '''
    def init_population(self, population_size=2):
        from configs import encode_config
        from tqdm import tqdm, trange

        self.max_cache_size = 1000000
        
        self.macs_cache = {}
        parents = []

        population, child_pool, macs_pool = [], [], []
        best_valids, best_infos = [], []

        for _ in trange(population_size, desc='Sample     '):
            sample = self.configs.sample()
            # ensure the generated sample is not the same as any in the pool cache
            while encode_config(sample) in self.child_pool_cache:
                sample = self.configs.sample()
            self.child_pool_cache.add(encode_config(sample))
            child_pool.append(sample)

        
        # Load data in this step for profiling macs in the next step
        # results = self.evaluate_without_cache(child_pool, self.cal_loss)

        results = self.evaluate_without_cache(child_pool, self.cal_loss)
        
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
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

    def run_evolution(self): 
        import matplotlib.pyplot as plt
        import pickle
        import copy

        self.child_pool_cache = set()  # this step resets child pool cache (avoid duplicated sampling)
        
        population_history = []
        population = self.init_population(self.population_size)  # will update child pool cache
        population_history.append(copy.deepcopy(population))

        for gen in range(self.gen_num):
            # import time
            # begin = time.time()

            # Variation (evluation for the offspring included)
            offspring = self.variation(population, self.p_crossover, self.p_mutation)  # will update child pool cache
            print([pop[1]['channels'] for pop in population])
            print([pop[1]['channels'] for pop in offspring])

            # P+Q
            population.extend(offspring)

            # import pdb; pdb.set_trace()
            # Environmental Selection
            population = self.environmental_selection(population, self.population_size)

            population_history.append(copy.deepcopy(population))

            # print("Time cost (1 generation):", time.time()-begin)
            # ### Plot
            # objs1 = [p[0]['obj'] for p in population] # obj(fid/mIoU)
            # objs2 = [p[-1] for p in population] # macs
            # # Save to file
            # plt.figure()
            # plt.plot(objs1, objs2, 'bo')
            # plt.title(str(gen))
            # save_dir = os.path.join(self.opt.log_dir, 'evolution', str(cur_training_steps))
            # os.makedirs(save_dir, exist_ok=True)
            # plt.savefig(os.path.join(save_dir, 'result{}.png'.format(gen)))
            # plt.close('all')

            # ### Save population
            # pop_pkl = os.path.join(save_dir, 'population{}.pkl'.format(gen))
            # # import pdb; pdb.set_trace()
            # with open(pop_pkl, "wb") as f:
            #     pickle.dump(population, f)

        evolved_pool = [sample[1] for sample in population]
        return evolved_pool, population_history
    


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
                # print(p.decision_variables, q.decision_variables, dominate(p, q))
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
        import numpy as np
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
        # import pdb; pdb.set_trace()
        import numpy as np
        pop_sorted = self.non_dominate_sorting(population)
        selected = []
        for front in pop_sorted:
            if len(selected) < n:
                if len(selected) + len(front) <= n:
                    selected.extend(front)
                else:
                    # select individuals according crowding distance here
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
        import numpy as np
        import copy
        from configs import encode_config

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
        results = self.evaluate_without_cache(child_pool, self.cal_loss)

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
    augment the child pool via variation without performance evaluation
    '''
    def augmentation_variation(self, population, p_crossover, p_mutation):
        import numpy as np
        import copy
        from configs import encode_config

        child_pool_set = set()
        # avoid generating sample same as that in population
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

        # Evaluate child pool
        # results = self.evaluate_without_cache(child_pool, self.cal_loss)

        for i in range(len(child_pool)):
            # Update macs
            # new_sample = child_pool[i]
            # macs = self.macs_cache.get(encode_config(new_sample))
            # if macs is None:
            #     macs, _ = self.profile(new_sample, verbose=False)
            #     if len(self.macs_cache) < self.max_cache_size:
            #         self.macs_cache[encode_config(new_sample)] = macs
            # macs_pool.append(macs) 

            offspring.append((0, child_pool[i], 0))

        return offspring


    def crossover_sample(self, sample1, sample2, p_crossover):
        import copy
        import random
        from configs import encode_config
        import numpy as np

        new_sample1 = copy.deepcopy(sample1)
        new_sample2 = copy.deepcopy(sample2)

        if np.random.random()<=p_crossover:
            for i in range(len(new_sample1['channels'])):
                choice = random.choice([0, 1])
                new_sample1['channels'][i] = sample2['channels'][i] if choice==1 else sample1['channels'][i]
                new_sample2['channels'][i] = sample1['channels'][i] if choice==1 else sample2['channels'][i]

        return new_sample1, new_sample2

    def mutate_sample(self, sample, mutate_prob):
        import copy
        import random
        from configs import encode_config

        new_sample = copy.deepcopy(sample)
        for i in range(len(new_sample['channels'])):
            if random.random() < mutate_prob:
                new_sample['channels'][i] = self.configs.sample_layer(i)
            
        return new_sample

    def profile(self, config=None, verbose=False):
        from torchprofile import profile_macs
        netG = self.netG_student
        if isinstance(netG, nn.DataParallel):
            netG = netG.module
        if config is not None:
            netG.configs = config
        with torch.no_grad():
            macs = profile_macs(netG, (self.real_A[:1],))
        params = 0
        for p in netG.parameters():
            params += p.numel()
        if verbose:
            print('MACs: %.3fG\tParams: %.3fM' % (macs / 1e9, params / 1e6), flush=True)
        return macs, params

    def calculate_eval_g_loss(self, config):
        # forward
        if isinstance(self.netG_student, nn.DataParallel):
            self.netG_student.module.configs = config
        else:
            self.netG_student.configs = config
        with torch.no_grad():
            self.Tfake_B = self.netG_teacher(self.real_A)
        self.Sfake_B = self.netG_student(self.real_A)

        # Generator
        if self.opt.dataset_mode == 'aligned':
            self.loss_G_recon = self.criterionRecon(self.Sfake_B, self.real_B) * self.opt.lambda_recon
            fake = torch.cat((self.real_A, self.Sfake_B), 1)
        else:
            self.loss_G_recon = self.criterionRecon(self.Sfake_B, self.Tfake_B) * self.opt.lambda_recon
            fake = self.Sfake_B
        pred_fake = self.netD(fake)
        self.loss_G_gan = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan
        if self.opt.lambda_distill > 0:
            self.loss_G_distill = self.calc_distill_loss() * self.opt.lambda_distill
        else:
            self.loss_G_distill = 0
        self.loss_G = self.loss_G_gan + self.loss_G_recon + self.loss_G_distill
    '''
    For evluating the child in EC 
    '''
    def evaluate_without_cache(self, child_pool, cal_loss=True):
        from models.spade_model import SPADEModel
        from configs import encode_config
        from metric import get_fid, get_coco_scores, get_cityscapes_mIoU
        
        opt = self.opt
        results = []
        for child in tqdm(child_pool, position=1, desc='Evaluate child pool   ', leave=False):
            cnt = 0
            result = {}
            fakes, names = [], []
            losses = []
            # if isinstance(self.model, SPADEModel):
            #     self.model.calibrate(child)
            
            for i, data_i in enumerate(self.eval_dataloader):
                if self.opt.dataset_mode == 'aligned':
                    self.set_input(data_i)
                else:
                    self.set_single_input(data_i)
                
                

                if cal_loss:
                    self.calculate_eval_g_loss(child)
                    losses.append(self.loss_G)
                else:
                    
                    # import time
                    # begin = time.time()
                    
                    self.test(child) # test model using a random sample
                    
                    # print("Time spent on feedforward a single batch:", time.time()-begin)
                    # import pdb; pdb.set_trace()

                    fakes.append(self.Sfake_B.cpu())
                for path in self.get_image_paths():
                    short_path = ntpath.basename(path)
                    name = os.path.splitext(short_path)[0]
                    names.append(name)
                cnt += 1  
                # # import pdb; pdb.set_trace()
                if cnt>=self.eval_cnt:  ###### Using single eval image
                    break

                
            # print("Time spent on feedforward (evaluate_without_cache_original):", time.time()-begin)

            # import pdb; pdb.set_trace()
            if cal_loss:
                loss_mean = torch.tensor(losses).mean().item()
                if self.inception_model is not None:
                    result['fid'] = loss_mean
                    result['obj'] = result['fid']
                if self.drn_model is not None:
                    result['mIoU'] = loss_mean
                    result['obj'] = -result['mIoU']  # minimization
                # if self.deeplabv2_model is not None:
                #     result['accu'] = torch.tensor(losses).mean()
            else:
                if self.inception_model is not None:
                    if 'horse2zebra' in self.opt.dataroot:
                        result['fid'] = get_fid(fakes, self.inception_model, self.npz, self.device,
                                                opt.batch_size, fast=True, tqdm_position=2)  # Fast FID evaluation
                    else:
                        result['fid'] = 0
                    result['obj'] = result['fid']
                if self.drn_model is not None:
                    
                    result['mIoU'] = get_cityscapes_mIoU(fakes, names, self.drn_model, self.device,
                                                            table_path=opt.table_path,
                                                            data_dir=opt.cityscapes_path, batch_size=2,  # 2 is the fastest
                                                            num_workers=opt.num_threads, tqdm_position=2)
                    
                    # import time
                    # begin = time.time()
                    # result['mIoU'] = get_cityscapes_mIoU(fakes, names, self.drn_model, self.device, table_path=opt.table_path, data_dir=opt.cityscapes_path, batch_size=2, num_workers=opt.num_threads, tqdm_position=2)
                    # print("Time spent on feedforward a single batch:", time.time()-begin)

                    # import pdb; pdb.set_trace()

                    result['obj'] = -result['mIoU']  # minimization
                if self.deeplabv2_model is not None:
                    torch.cuda.empty_cache()
                    result['accu'], result['mIoU'] = get_coco_scores(fakes, names, self.deeplabv2_model, self.device,
                                                                        opt.dataroot, 1, num_workers=0, tqdm_position=2)
                # print(result)
            # print("Time spent (evaluate_without_cache_original):", time.time()-begin)
            results.append(result)
        return results

    def evaluate_without_cache_new(self, child_pool, cal_loss=True):        
        opt = self.opt
        results = []
        for child in tqdm(child_pool, position=1, desc='Evaluate child pool   ', leave=False):
            cnt = 0
            result = {}
            fakes, names = [], []
            losses = []
            # if isinstance(self.model, SPADEModel):
            #     self.model.calibrate(child)
            config_str = "_".join([str(x) for x in child['channels']])

            # import time
            # begin = time.time()

            fid, macs, params = test_fid(config_str, self.netG_student, self.opt.input_nc, self.opt.output_nc, self.opt.student_ngf, self.inception_model, self.device)
            
            # print("Time spent (Test FID):", time.time()-begin)

            result['fid'] = fid
            result['obj'] = result['fid']
            results.append(result)
            
            cnt += 1  
            # import pdb; pdb.set_trace()
            if cnt>=self.eval_cnt:  ###### Using single eval image
                break

        return results
    

    def forward(self, config):
        if isinstance(self.netG_student, nn.DataParallel):
            self.netG_student.module.configs = config
        else:
            self.netG_student.configs = config
        with torch.no_grad():
            self.Tfake_B = self.netG_teacher(self.real_A)
        self.Sfake_B = self.netG_student(self.real_A)

    def calc_distill_loss(self):
        losses = []
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, SuperConv2d)
            n = self.mapping_layers[i]
            netA_replicas = replicate(netA, self.gpu_ids)
            kwargs = tuple([{'config': {'channel': netA.out_channels}} for idx in self.gpu_ids])
            Sacts = parallel_apply(netA_replicas,
                                   tuple([self.Sacts[key] for key in sorted(self.Sacts.keys()) if n in key]), kwargs)
            Tacts = [self.Tacts[key] for key in sorted(self.Tacts.keys()) if n in key]
            loss = [F.mse_loss(Sact, Tact) for Sact, Tact in zip(Sacts, Tacts)]
            loss = gather(loss, self.gpu_ids[0]).sum()
            setattr(self, 'loss_G_distill%d' % i, loss)
            losses.append(loss)
        return sum(losses)

    def optimize_parameters(self, steps):
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()   

        # Sample from the population
        if self.evolution_sampling_flag:
            config = random.choice(self.evolved_pool)
        # Uniformly sampling
        else:
            config = self.configs.sample()

        # print(config)
        # import pdb; pdb.set_trace()
        self.forward(config=config)
        util.set_requires_grad(self.netD, True)
        self.backward_D()
        util.set_requires_grad(self.netD, False)
        self.backward_G()
        self.optimizer_D.step()
        self.optimizer_G.step()

    def evaluate_model(self, step):
        ret = {}
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG_student.eval()
        if self.opt.eval_mode == 'both':
            settings = ('largest', 'smallest')
        else:
            settings = (self.opt.eval_mode,)
        for config_name in settings:
            config = self.configs(config_name)
            fakes, names = [], []
            cnt = 0
            for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
                if self.opt.dataset_mode == 'aligned':
                    self.set_input(data_i)
                else:
                    self.set_single_input(data_i)
                self.test(config)
                fakes.append(self.Sfake_B.cpu())
                for j in range(len(self.image_paths)):
                    short_path = ntpath.basename(self.image_paths[j])
                    name = os.path.splitext(short_path)[0]
                    names.append(name)
                    if i < 10:
                        Sfake_im = util.tensor2im(self.Sfake_B[j])
                        real_im = util.tensor2im(self.real_B[j])
                        Tfake_im = util.tensor2im(self.Tfake_B[j])
                        util.save_image(real_im, os.path.join(save_dir, 'real', '%s.png' % name), create_dir=True)
                        util.save_image(Sfake_im, os.path.join(save_dir, 'Sfake_%s' % config_name, '%s.png' % name),
                                        create_dir=True)
                        util.save_image(Tfake_im, os.path.join(save_dir, 'Tfake', '%s.png' % name), create_dir=True)
                        if self.opt.dataset_mode == 'aligned':
                            input_im = util.tensor2im(self.real_A[j])
                            util.save_image(input_im, os.path.join(save_dir, 'input', '%s.png') % name, create_dir=True)
                    cnt += 1

            fid = get_fid(fakes, self.inception_model, self.npz, device=self.device,
                          batch_size=self.opt.eval_batch_size, tqdm_position=2)
            if fid < getattr(self, 'best_fid_%s' % config_name):
                self.is_best = True
                setattr(self, 'best_fid_%s' % config_name, fid)
            fids = getattr(self, 'fids_%s' % config_name)
            fids.append(fid)
            if len(fids) > 3:
                fids.pop(0)

            ret['metric/fid_%s' % config_name] = fid
            ret['metric/fid_%s-mean' % config_name] = sum(getattr(self, 'fids_%s' % config_name)) / len(
                getattr(self, 'fids_%s' % config_name))
            ret['metric/fid_%s-best' % config_name] = getattr(self, 'best_fid_%s' % config_name)

            if 'cityscapes' in self.opt.dataroot:
                mIoU = get_cityscapes_mIoU(fakes, names, self.drn_model, self.device,
                                           table_path=self.opt.table_path,
                                           data_dir=self.opt.cityscapes_path,
                                           batch_size=self.opt.eval_batch_size,
                                           num_workers=self.opt.num_threads, tqdm_position=2)
                if mIoU > getattr(self, 'best_mIoU_%s' % config_name):
                    self.is_best = True
                    setattr(self, 'best_mIoU_%s' % config_name, mIoU)
                mIoUs = getattr(self, 'mIoUs_%s' % config_name)
                mIoUs.append(mIoU)
                if len(mIoUs) > 3:
                    mIoUs.pop(0)
                ret['metric/mIoU_%s' % config_name] = mIoU
                ret['metric/mIoU_%s-mean' % config_name] = sum(getattr(self, 'mIoUs_%s' % config_name)) / len(
                    getattr(self, 'mIoUs_%s' % config_name))
                ret['metric/mIoU_%s-best' % config_name] = getattr(self, 'best_mIoU_%s' % config_name)

        self.netG_student.train()
        return ret

    def test(self, config):
        with torch.no_grad():
            self.forward(config)

    def load_networks(self, verbose=True):
        super(ResnetSupernet, self).load_networks()

    def save_networks(self, epoch):
        super(ResnetSupernet, self).save_networks(epoch)
    
    
    def evolution(self, epoch):
        # import pdb; pdb.set_trace()
        # with open("logs/cycle_gan/horse2zebra_fast/supernet_continue_evolution0621/evolution/100/population_3.pkl", 'rb') as f: population = pickle.load(f)
        #     population_constrain = [ind for ind in population if ind[-1]<2.6e9]
        #     population_sorted = sorted(population_constrain, key=lambda x:x[0]['fid'])
        #     config_str = "_".join([str(x) for x in population_sorted[0][1]['channels']])
        #     fid, macs, params = test_fid(config_str, self.netG_student, self.opt.input_nc, self.opt.output_nc, self.opt.student_ngf, self.inception_model, self.device)
        # MACs: 2.540G     Params: 0.295M
        # fid score: 57.65
        # '32_24_16_16_48_64_16_24'
        
          
        # import pdb; pdb.set_trace()
        #####################  Obtain augmented solution set #################################
        # Test augmented population
#         import pickle
#         with open("logs/cycle_gan/horse2zebra_fast/supernet_continue_evolution0621/evolution/100/population_1.pkl", 'rb') as f: population = pickle.load(f)
#         self.child_pool_cache = set()
#         self.macs_cache = {}
#         self.max_cache_size = 1000000
#         population_evaluated = self.variation(population, self.p_crossover, self.p_mutation)
#         ppp = self.variation(population, 1, 0.1)
        
#         with open("logs/cycle_gan/horse2zebra_fast/supernet_continue_evolution0621/evolution/100/population_1_augmented.pkl", "wb") as f:             pickle.dump(ppp, f)
    
#         import pdb; pdb.set_trace()
        #####################  Obtain augmented solution set #################################
        
        # import pdb; pdb.set_trace()
        if self.opt.config_str is None:  # Not in fine-tuning mode (no config_str specified)
            import copy
            if epoch>=self.opt.warmup_epochs:
                # iteratively sampling from pool and uniform distribution
                if hasattr(self, "evolved_pool"):
                    self.evolution_sampling_flag = not self.evolution_sampling_flag # display evolution sampling for the next iteration
                    # Randomly augment samples based on the evolved pool via variation operation (without evaluation)
                    if self.evolution_sampling_flag and hasattr(self, "population_history"):
                        offspring_unevaluated = self.augmentation_variation(self.population_history[-1], self.p_crossover, self.p_mutation)
                        augmented_pool_unevaluated = [offspring[1] for offspring in offspring_unevaluated]
                        self.evolved_pool = self.evolved_pool_evaluated + augmented_pool_unevaluated  # combine two lists
                        
                # start evolution and save population history
                if epoch % self.opt.evolution_epoch_freq == 0:
                    # start evolution  
                    if epoch != (self.opt.nepochs + self.opt.nepochs_decay):
                        self.evolved_pool, self.population_history = self.run_evolution() # reset sample cache
                    else:
                        # last epoch
                        self.population_size = 50
                        self.gen_num = 10
                        self.evolved_pool, self.population_history = self.run_evolution() # reset sample cache
                    self.evolved_pool_evaluated = copy.deepcopy(self.evolved_pool)
                    self.evolution_sampling_flag = True
                    
                    ### Plot
                    import matplotlib.pyplot as plt
                    import pickle
                    
                    for idx, population in enumerate(self.population_history):
                        objs1 = [p[0]['obj'] for p in population] # obj(fid/mIoU)
                        objs2 = [p[-1] for p in population] # macs
                        # Save to file
                        plt.figure()
                        plt.plot(objs1, objs2, 'bo')
                        plt.title(str(epoch))
                        save_dir = os.path.join(self.opt.log_dir, 'evolution', str(epoch))
                        os.makedirs(save_dir, exist_ok=True)
                        plt.savefig(os.path.join(save_dir, 'result_{}.png'.format(idx)))
                        plt.close('all')

                        ### Save population
                        pop_pkl = os.path.join(save_dir, 'population_{}.pkl'.format(idx))

                        # import pdb; pdb.set_trace()
                        with open(pop_pkl, "wb") as f:
                            pickle.dump(population, f)

                    # Test FID
                    '''
                    Test the best fid after evolution
                    # def test_fid(restore_student_G_path=opt.restore_student_G_path, config_str, input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.student_ngf, inception_model=model.inception_model, device=model.device):
                    '''
                    # import pdb; pdb.set_trace()
                    # with open("logs/cycle_gan/horse2zebra_fast/supernet_continue_evolution0621/evolution/105/population_3.pkl", 'rb') as f: population = pickle.load(f)
                    population_constrain = [ind for ind in population if ind[-1]<2.6e9]
                    population_sorted = sorted(population_constrain, key=lambda x:x[0]['fid'])
                    config_str = "_".join([str(x) for x in population_sorted[0][1]['channels']])
                    # "16_24_16_16_48_64_24_16"
                    fid, macs, params = test_fid(config_str, self.netG_student, self.opt.input_nc, self.opt.output_nc, self.opt.student_ngf, self.inception_model, self.device)
                    
