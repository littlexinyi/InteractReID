export CUDA_VISIBLE_DEVICES=0

#-------------------------Market-Sketch-1K-------------------
python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/Market_finetune.yaml" --ckpt_type 'finetune_train' --dataset 'MaSk1K' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch20_bz32' --pretrain_data 'Tri-PEDES' --mapping_type 'MLP' --n_layer 1 --middle_dim 768

#--------------------------PKUSketch---------------------
python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/PKU_finetune.yaml" --dataset 'PKUSketch' --ckpt_type 'finetune_train' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16' --pretrain_data 'Tri-PEDES' --trial 1 --mapping_type 'MLP' --n_layer 1 --middle_dim 768

python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/PKU_finetune.yaml" --dataset 'PKUSketch' --ckpt_type 'finetune_train' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16' --pretrain_data 'Tri-PEDES' --trial 2 --mapping_type 'MLP' --n_layer 1 --middle_dim 768

python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/PKU_finetune.yaml" --dataset 'PKUSketch' --ckpt_type 'finetune_train' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16' --pretrain_data 'Tri-PEDES' --trial 3 --mapping_type 'MLP' --n_layer 1 --middle_dim 768

python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/PKU_finetune.yaml" --dataset 'PKUSketch' --ckpt_type 'finetune_train' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16' --pretrain_data 'Tri-PEDES' --trial 4 --mapping_type 'MLP' --n_layer 1 --middle_dim 768

python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/PKU_finetune.yaml" --dataset 'PKUSketch' --ckpt_type 'finetune_train' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16' --pretrain_data 'Tri-PEDES' --trial 5 --mapping_type 'MLP' --n_layer 1 --middle_dim 768

python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/PKU_finetune.yaml" --dataset 'PKUSketch' --ckpt_type 'finetune_train' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16' --pretrain_data 'Tri-PEDES' --trial 6 --mapping_type 'MLP' --n_layer 1 --middle_dim 768

python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/PKU_finetune.yaml" --dataset 'PKUSketch' --ckpt_type 'finetune_train' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16' --pretrain_data 'Tri-PEDES' --trial 7 --mapping_type 'MLP' --n_layer 1 --middle_dim 768

python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/PKU_finetune.yaml" --dataset 'PKUSketch' --ckpt_type 'finetune_train' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16' --pretrain_data 'Tri-PEDES' --trial 8 --mapping_type 'MLP' --n_layer 1 --middle_dim 768

python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/PKU_finetune.yaml" --dataset 'PKUSketch' --ckpt_type 'finetune_train' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16' --pretrain_data 'Tri-PEDES' --trial 9 --mapping_type 'MLP' --n_layer 1 --middle_dim 768

python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=1 --use_env pseudo_token_learning.py --config_path "config/PKU_finetune.yaml" --dataset 'PKUSketch' --ckpt_type 'finetune_train' --exp_name 'PreTriPEDES5_Shared_FTViT16_1MLPdim768_epoch40_bz16' --pretrain_data 'Tri-PEDES' --trial 10 --mapping_type 'MLP' --n_layer 1 --middle_dim 768

