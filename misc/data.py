import json
import os
import random

import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms

from misc.caption_dataset import *

from misc.utils import is_using_distributed, Random_ID


def get_self_supervised_augmentation(img_size):
    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[.1, 2.]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    aug = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.), antialias=True),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    return aug


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def build_pedes_data(config):
    size = config.experiment.input_resolution
    if isinstance(size, int):
        size = (size, size)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    val_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        normalize
    ])

    rand_from = [
        transforms.ColorJitter(.1, .1, .1, 0),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(size, (0.9, 1.0), antialias=True),
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.10, 0.20)),
    ]
    aug = Choose(rand_from, size)
    aug_ss = get_self_supervised_augmentation(size)

    if(config.data.dataset == 'MaSk1K'):
        train_dataset = MaSk1K_train(config.data.anno_dir, aug)
        test_gallery_dataset = Mask1K_test_img(config.data.anno_dir, val_transform)
        test_query_dataset = Mask1K_test_sk(config.data.anno_dir, val_transform)

    elif(config.data.dataset == 'PKUSketch'):
        
        # train_id, test_id = Random_ID(config.data.anno_dir, seed = config.data.seed)       #每次运行程序随机生成两个ID 序列      
        root = config.data.anno_dir
        trial = config.data.trial
        train_visible_path = root+'idx/train_visible_{}.txt'.format(trial)
        train_sketch_path = root+'idx/train_sketch_{}.txt'.format(trial)
        test_visible_path = root+'idx/test_visible_{}.txt'.format(trial)
        test_sketch_path = root+'idx/test_sketch_{}.txt'.format(trial)        
        train_file_list = open(train_sketch_path, 'rt').read().splitlines()
        train_id = [int(s.split(' ')[1]) for s in train_file_list]
        
        test_file_list = open(test_sketch_path, 'rt').read().splitlines()
        test_id = [int(s.split(' ')[1]) for s in test_file_list]

        train_dataset = PKUSketch_Train(root, train_id, aug)
        test_gallery_dataset = PKUSketch_Test_img(root, test_id, val_transform)
        test_query_dataset = PKUSketch_Test_query(root, test_id, val_transform)

    elif(config.data.dataset == 'CUHK-PEDES'):      
        train_dataset = ps_train_dataset(config.data.anno_dir1, aug, aug_ss, split='train', max_words=77)
        test_gallery_dataset = ps_eval_img_dataset(config.data.anno_dir1, val_transform, split='test', max_words=77)
        test_query_dataset = ps_eval_text_dataset(config.data.anno_dir1, val_transform, split='test', max_words=77)
        
    elif(config.data.dataset == 'ICFG-PEDES'):      
        train_dataset = ps_train_dataset(config.data.anno_dir2, aug, aug_ss, split='train', max_words=77)
        test_gallery_dataset = ps_eval_img_dataset(config.data.anno_dir2, val_transform, split='test', max_words=77)
        test_query_dataset = ps_eval_text_dataset(config.data.anno_dir2, val_transform, split='test', max_words=77)

    elif(config.data.dataset == 'RSTPReid'):      
        train_dataset = ps_train_dataset(config.data.anno_dir3, aug, aug_ss, split='train', max_words=77)
        test_gallery_dataset = ps_eval_img_dataset(config.data.anno_dir3, val_transform, split='test', max_words=77)
        test_query_dataset = ps_eval_text_dataset(config.data.anno_dir3, val_transform, split='test', max_words=77)

    elif(config.data.dataset == 'Tri-PEDES'):
        train_dataset = pedes_train_dataset(config.data.anno_dir1, config.data.anno_dir2, config.data.anno_dir3, aug, aug_ss, split='train', max_words=77)

        test_gallery_dataset_cuhk = ps_eval_img_dataset(config.data.anno_dir1, val_transform, split='test', max_words=77)

        test_query_dataset_cuhk = ps_eval_text_dataset(config.data.anno_dir1, val_transform, split='test', max_words=77)   

        test_gallery_dataset_icfg = ps_eval_img_dataset(config.data.anno_dir2, val_transform, split='test', max_words=77)

        test_query_dataset_icfg = ps_eval_text_dataset(config.data.anno_dir2, val_transform, split='test', max_words=77)  

        test_gallery_dataset_rstp = ps_eval_img_dataset(config.data.anno_dir3, val_transform, split='test', max_words=77)

        test_query_dataset_rstp = ps_eval_text_dataset(config.data.anno_dir3, val_transform, split='test', max_words=77)   

    if is_using_distributed():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    test_sampler = None

    config_data = config.data
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config_data.batch_size,
        shuffle=train_sampler is None,
        num_workers=config_data.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    if(config.data.dataset == 'Tri-PEDES'):

        test_gallery_loader_cuhk = DataLoader(
            dataset=test_gallery_dataset_cuhk,
            batch_size=config_data.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
        )
        
        test_query_loader_cuhk = DataLoader(
            dataset=test_query_dataset_cuhk,
            batch_size=config_data.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
        )

        test_gallery_loader_icfg = DataLoader(
            dataset=test_gallery_dataset_icfg,
            batch_size=config_data.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
        )
        
        test_query_loader_icfg = DataLoader(
            dataset=test_query_dataset_icfg,
            batch_size=config_data.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
        )

        test_gallery_loader_rstp = DataLoader(
            dataset=test_gallery_dataset_rstp,
            batch_size=config_data.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
        )
        
        test_query_loader_rstp = DataLoader(
            dataset=test_query_dataset_rstp,
            batch_size=config_data.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
        )

        return {
            'train_loader': train_loader,
            'train_sampler': train_sampler,
            'test_gallery_loader_rstp': test_gallery_loader_rstp,
            'test_query_loader_rstp': test_query_loader_rstp,

            'test_gallery_loader_icfg': test_gallery_loader_icfg,
            'test_query_loader_icfg': test_query_loader_icfg,  

            'test_gallery_loader_cuhk': test_gallery_loader_cuhk,
            'test_query_loader_cuhk': test_query_loader_cuhk,     
        }
    
    else:
        test_gallery_loader = DataLoader(
            dataset=test_gallery_dataset,
            batch_size=config_data.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
        )
        
        test_query_loader = DataLoader(
            dataset=test_query_dataset,
            batch_size=config_data.test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
        )

        return {
            'train_loader': train_loader,
            'train_sampler': train_sampler,
            'test_gallery_loader': test_gallery_loader,
            'test_query_loader': test_query_loader,
        }
