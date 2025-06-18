export CUDA_VISIBLE_DEVICES=0

#--------------------------Tri-PEDES---------------------
# python model_directly_test.py --config_path "config/TriPEDES_pretrain.yaml" --ckpt_type 'direct_test' --dataset 'Tri-PEDES' --exp_name 'Shared_Vit16_trimodal_token77_bz240_epoch5'

#-------------------------Market-Sketch-1K-------------------
python model_directly_test.py --config_path "config/Market_finetune.yaml" --ckpt_type 'direct_test' --dataset 'MaSk1K' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch20_bz32'

#--------------------------PKUSketch---------------------
# python model_directly_test.py --config_path "config/PKU_finetune.yaml" --ckpt_type 'direct_test' --dataset 'PKUSketch' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16'

