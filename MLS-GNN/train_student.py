import argparse
import itertools
from distill_process import raw_experiment
from collections import defaultdict, namedtuple
from pre_configs import predefined_configs
import torch
import warnings
import random
import numpy as np


warnings.simplefilter(action='ignore', category=UserWarning)



def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--student', type=str, default='MLS-GCN', help='Student Model')
    parser.add_argument('--distill', action='store_false', default=True, help='Distill or not')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--labelrate', type=int, default=20, help='label rate')
    parser.add_argument('--valrate', type=float,default=30,  help='val rate')
    parser.add_argument('--base_test', default=999, type=int, help='best_score')
    ##
    parser.add_argument('--max_epoch', default=200, type=int, help='epoch')
    parser.add_argument('--patience', default=50, type=int, help='early_stop')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optim')
    parser.add_argument('--model_name', default='MLS-GNN', type=str, help='student_name')
    parser.add_argument('--division_seed', default=0, type=int, help='division')
    parser.add_argument('--teacher_seed', default=61, type=int, help='learn from different teacher seed')
    parser.add_argument('--seed', default=2, type=int, help='student seed')
    parser.add_argument('--teacher_amounts', default=2, type=int, help='teachers')
    return parser.parse_args()



def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    print()
    return itertools.starmap(Variant, itertools.product(*items.values()))



if __name__ == '__main__':

    args = arg_parse(argparse.ArgumentParser())
    configs = dict(args.__dict__, **predefined_configs[args.teacher][args.dataset])
    configs['device'] = torch.device("cuda:0")
    SEED = configs['seed']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    variants = list(gen_variants(dataset=[configs['dataset']],
                                 model=[configs['model_name']]
                                 ))
    results_dict = defaultdict(list)
    print(configs)
    print(variants)

    raw_experiment(configs)


