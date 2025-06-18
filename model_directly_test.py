import os
import random
import time
from pathlib import Path

import torch

from misc.build import load_checkpoint
from misc.data import build_pedes_data
from misc.eval import fintune_val, pretrain_val
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
from model.tbps_model import model_with_textual_inversion, clip_vitb
from options import get_args
from misc.log import get_logger
from misc.utils import AverageMeter

def run(config, logger):
    logger.info(config)
    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)

    if(config.data.dataset == 'Tri-PEDES'):    
        model = clip_vitb(config, num_classes)
    else:
        model = model_with_textual_inversion(config, num_classes)

    model.cuda()
    model, load_result = load_checkpoint(model, config)

    for param in model.parameters():
        param.requires_grad_(False)  

    if(config.data.dataset == 'Tri-PEDES'):     

        table_cuhk, rank_1_cuhk = pretrain_val(model, dataloader['test_gallery_loader_cuhk'],dataloader['test_query_loader_cuhk'], config)
        logger.info('\n' +"------CUHK-Validation------"+'\n' + str(table_cuhk))

        table_icfg, rank_1_icfg = pretrain_val(model, dataloader['test_gallery_loader_icfg'],dataloader['test_query_loader_icfg'], config)
        logger.info('\n' +"------ICFG-Validation------"+'\n' + str(table_icfg))

        table_rstp, rank_1_rstp = pretrain_val(model, dataloader['test_gallery_loader_rstp'],dataloader['test_query_loader_rstp'], config)
        logger.info('\n' +"------RSTP-Validation------"+'\n' + str(table_rstp))
        table = 0
    else:
        table, rank_1 = fintune_val(model, dataloader['test_gallery_loader'], dataloader['test_query_loader'], config)
        logger.info('\n' +"------{}-Validation------".format(config.data.dataset)+'\n' + str(table))

    torch.cuda.empty_cache()

    return table
    
if __name__ == '__main__':

    args = get_args()
    config_path = args.config_path
    config = parse_config(config_path)
    config.model.ckpt_type = args.ckpt_type

    if(config.data.dataset == 'Tri-PEDES'):
        config.model.saved_path = os.path.join(config.model.saved_path, config.data.dataset, config.model.exp_name)
        log_path = os.path.join(config.model.saved_path, 'test_logs/')    

    elif(config.data.dataset == 'PKUSketch'):
        config.model.exp_name = args.exp_name
        config.model.mapping_type = args.mapping_type
        config.model.middle_dim = args.middle_dim
        if(config.model.mapping_type == 'MLP'):
            config.model.n_layer = args.n_layer
        # config.data.trial = args.trial
        config.model.output_path = os.path.join(config.model.output_path, config.data.dataset, "pretrained_{}".format(config.data.pretrain_data), config.model.exp_name)
        log_path = os.path.join(config.model.output_path, 'test_logs/')

    elif(config.data.dataset == 'MaSk1K'):
        config.model.exp_name = args.exp_name
        config.model.output_path = os.path.join(config.model.output_path, config.data.dataset, "pretrained_{}".format(config.data.pretrain_data), config.model.exp_name)
        log_path = os.path.join(config.model.output_path, 'test_logs/')

    init_distributed_mode(config)
    set_seed(config)
    os.makedirs(log_path, exist_ok=True)
    logger = get_logger(log_path)
    
    if(config.data.dataset == 'PKUSketch'):    
        meters = {
            "interactive_R1": AverageMeter(),
            "interactive_R5": AverageMeter(),  
            "interactive_R10": AverageMeter(),     
            "interactive_mAP": AverageMeter(),
            "interactive_mINP": AverageMeter(),
            "sketchonly_R1": AverageMeter(),
            "sketchonly_R5": AverageMeter(),  
            "sketchonly_R10": AverageMeter(),     
            "sketchonly_mAP": AverageMeter(),
            "sketchonly_mINP": AverageMeter(),
        } 
        for meter in meters.values():
            meter.reset()   
        Table_list = []
        model_path = config.model.output_path
        for trial in range(1, 11):
            logger.info("-------Trial-------:{}".format(trial))
            config.data.trial = trial
            config.model.output_path = os.path.join(model_path, "Trial_{}".format(trial))
            table_i = run(config, logger)
            Table_list.append(table_i)

        for table_trial in Table_list:
            for row in table_trial.rows:
                print(row)
                if(row[0] == 'interactive_retrieval'):
                    meters["interactive_R1"].update(row[1])
                    meters["interactive_R5"].update(row[2])
                    meters["interactive_R10"].update(row[3])
                    meters["interactive_mAP"].update(row[4])
                    meters["interactive_mINP"].update(row[5])
                if(row[0] == 'sketchonly_retrieval'):
                    meters["sketchonly_R1"].update(row[1])
                    meters["sketchonly_R5"].update(row[2])
                    meters["sketchonly_R10"].update(row[3])
                    meters["sketchonly_mAP"].update(row[4])
                    meters["sketchonly_mINP"].update(row[5])
        for k, v in meters.items():
            logger.info("{}: {} \n".format(k, v.average()))

    else:
        table = run(config, logger)


