import argparse


def get_args():
    parser = argparse.ArgumentParser(description="InteractReID Args")
    ######################## mode ########################
    # parser.add_argument("--simplified", default=False, action='store_true')
    parser.add_argument("--config_path", default='config/TriPEDES_pretrain_simplified.yaml')

    parser.add_argument("--dataset", default='Tri-PEDES')
    
    parser.add_argument("--exp_name", default='PreCUHK_Shared_ViT16_FTALL_epoch40_bz16_trial1')

    parser.add_argument("--trial", default=1, type=int)

    parser.add_argument("--pretrain_data", default='Tri-PEDES')

    parser.add_argument("--ckpt_type", default='pretrain')

    parser.add_argument("--mapping_type", default='MLP')

    parser.add_argument("--n_layer", default=1, type=int)

    parser.add_argument("--middle_dim", default=768, type=int)



    args = parser.parse_args()

    return args