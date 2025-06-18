from pathlib import Path

from easydict import EasyDict

import yaml
import os
import torch
import numpy as np
import random
import torch.distributed as dist

import os



def parse_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    return config


def is_using_distributed():
    return False if 'LOCAL_RANK' not in os.environ else True


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_master():
    return not is_using_distributed() or get_rank() == 0


def wandb_record():
    if not 'WANDB_PROJECT' in os.environ:
        return False
    return not is_using_distributed() or get_rank() == 0


def init_distributed_mode(config):
    if is_using_distributed():
        config.distributed.rank = int(os.environ['RANK'])
        config.distributed.world_size = int(os.environ['WORLD_SIZE'])
        config.distributed.local_rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend=config.distributed.backend,
                                             init_method=config.distributed.url)
        used_for_printing(get_rank() == 0)

    if torch.cuda.is_available():
        if is_using_distributed():
            device = f'cuda:{get_rank()}'
        else:
            device = f'cuda:{d}' if str(d := config.device).isdigit() else d
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    config.device = device


def used_for_printing(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_seed(config):
    seed = config.misc.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

import glob

def read_txt_file(file_path):
    lines = []
    with open(file_path) as f:
        lines.append(f.read().splitlines() )
    f.close()
    lines = np.hstack(lines)
    return lines

def Random_ID(path, seed=1):
    data_dir = path
    style_dir = data_dir + '/styleAnnotation/'
    style_list = glob.glob(style_dir + '*.txt')
    style_list.sort()
    sample_ind = {}
    for style_path in style_list:
        style_clc = style_path.split('/')[-1].split('_')[0]
        lines = read_txt_file(style_path)
        index = [int(line) for line in lines]
        sample_ind[style_clc] = index
    train_id = []
    test_id = []
    split_pos = [34, 15, 60, 25, 16] # given by 'Cross-Domain Adversarial Feature Learning for Sketch Re-identification'
    for style_clc, split in zip(sample_ind, split_pos):
        all_ind = np.random.RandomState(seed=seed).permutation(sample_ind[style_clc])
        train_id += list(all_ind[:split])
        test_id += list(all_ind[split:])
    return train_id, test_id



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def average(self):
        return self.avg

def unfreeze_ln(m):
    if isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(True)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(True)