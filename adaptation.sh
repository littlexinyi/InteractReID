export CUDA_VISIBLE_DEVICES=0,1,2,3 

#--------------------------Tri-PEDES---------------------
python -m torch.distributed.launch --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
--nnodes=1 --nproc_per_node=4 --use_env knowledge_adaptation.py --config_path "config/TriPEDES_pretrain.yaml" --ckpt_type 'pretrain' --dataset "Tri-PEDES"
