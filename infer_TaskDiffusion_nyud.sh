RANDOM=$$
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main.py --config_exp './configs/nyud/nyud_vitLp16.yml' \
    --run_mode infer \
    --trained_model taskdiffusion_nyud_vitl.pth.tar