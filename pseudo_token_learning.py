import os
import random
import time
from pathlib import Path


import torch
from misc.utils import parse_config, init_distributed_mode, \
          set_seed, is_master, is_using_distributed, AverageMeter, unfreeze_ln
from misc.log import get_logger
from misc.build import load_checkpoint, cosine_scheduler, build_optimizer, load_checkpoint_vit
from misc.eval import fintune_val

from misc.data import build_pedes_data
from model.tbps_model import model_with_textual_inversion
from options import get_args


def run(config, logger):
    if is_master():
        logger.info(f"Config: {config}")

    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)

    meters = {
        "loss": AverageMeter(),
        "inversion_loss": AverageMeter(),
        "token_align_loss": AverageMeter(),  
    }
    best_rank_1 = 0.0
    best_epoch = 0

    # model
    model = model_with_textual_inversion(config, num_classes)
    model.cuda()

    #NOTE ckpt_type: finetune_train  shared vision encoder for img and sketch
    model.tbps_clip, _ = load_checkpoint(model.tbps_clip, config)

    #NOTE frozen text encoder, FT all visual encoder param & mapping network
    for param in model.tbps_clip.encode_text.parameters():
        param.requires_grad_(False)    
    
    if is_master():
        enabled = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logger.info(f"Parameters to be updated: {enabled}")
        total_param = sum(p.numel() for p in model.parameters())  / 1000000
        trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)  / 1000000
        logger.info(f"Model Total Parameters: {total_param:.2f} M, Model Trainable Parameters: {trainable_param:.2f} M")       
    
    if is_using_distributed():
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device],
        #                                                   find_unused_parameters=True)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None,
                                                          find_unused_parameters=True)

    # schedule
    config.schedule.niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(config)

    # optimizer
    optimizer = build_optimizer(config, model)

    # train
    it = 0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.schedule.epoch):
        if is_using_distributed():
            dataloader['train_sampler'].set_epoch(epoch)

        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        # Train
        for i, batch in enumerate(train_loader):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[it] * param_group['ratio']

            if epoch == 0:
                alpha = config.model.softlabel_ratio * min(1.0, i / len(train_loader))
            else:
                alpha = config.model.softlabel_ratio


            with torch.autocast(device_type='cuda'):
                ret = model(batch, alpha)
                loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['image'].shape[0]
            meters['loss'].update(loss.item(), batch_size)
            meters['inversion_loss'].update(ret.get('inversion_loss', 0), batch_size)
            meters['token_align_loss'].update(ret.get('token_align_loss', 0), batch_size)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            optimizer.zero_grad()
            it += 1

            if (i + 1) % config.log.print_period == 0:
                info_str = f"Epoch[{epoch + 1}] Iteration[{i + 1}/{len(train_loader)}]"
                # log loss
                for k, v in meters.items():
                    if v.val != 0:
                        info_str += f", {k}: {v.val:.4f}"
                info_str += f", Base Lr: {param_group['lr']:.2e}"
                if is_master():
                    logger.info(info_str)
        
        # Evaluation
        if is_master():
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (i + 1)
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                  .format(epoch + 1, time_per_batch, train_loader.batch_size / time_per_batch))

            table, rank_1 = fintune_val(model.module, dataloader['test_gallery_loader'], dataloader['test_query_loader'], config)
            logger.info('\n' + str(table))

            torch.cuda.empty_cache()
            if best_rank_1 < rank_1:
                best_rank_1 = rank_1
                best_epoch = epoch

                save_obj = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                }
                torch.save(save_obj, os.path.join(config.model.output_path, 'checkpoint_best.pth'))

            logger.info(f"best R@1: {best_rank_1} at epoch {best_epoch + 1}")


if __name__ == '__main__':

    args = get_args()
    config_path = args.config_path
    config = parse_config(config_path)

    config.data.dataset = args.dataset
    config.model.exp_name = args.exp_name
    config.data.pretrain_data = args.pretrain_data
    config.model.ckpt_type = args.ckpt_type
    config.model.mapping_type = args.mapping_type
    config.model.middle_dim = args.middle_dim
    
    if(config.model.mapping_type == 'MLP'):
        config.model.n_layer = args.n_layer

    config.model.saved_path = os.path.join(config.model.saved_path, config.data.pretrain_data, 'Shared_Vit16_trimodal_token77_bz240_epoch5')

    config.model.output_path = os.path.join(config.model.output_path, config.data.dataset, "pretrained_{}".format(config.data.pretrain_data), config.model.exp_name)

    if(config.data.dataset == 'PKUSketch'):
        config.data.trial = args.trial    
        config.model.output_path = os.path.join(config.model.output_path, "Trial_{}".format(config.data.trial))

    Path(config.model.saved_path).mkdir(parents=True, exist_ok=True)
    Path(config.model.output_path).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(config.model.output_path, 'logs/')
    os.makedirs(log_path, exist_ok=True)

    init_distributed_mode(config)
    set_seed(config)
    logger = get_logger(log_path)

    run(config, logger)
