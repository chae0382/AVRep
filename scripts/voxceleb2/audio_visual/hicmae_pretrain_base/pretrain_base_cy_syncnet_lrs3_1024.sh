#!/bin/zsh

dataset="lrs3"
OUTPUT_DIR="./results/${dataset}_1024"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi

# Set the path to pre-training dataset.
DATA_PATH="./preprocess/lrs3_2d_pretrain_b_1024.csv"
DATA_ROOT=$1

# batch_size can be adjusted according to number of GPUs
# this script is for 4 GPUs (1 nodes x 4 GPUs)

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
        --master_port 11145 \
        run_mae_pretraining_av_cy_syncnet.py \
        --data_path ${DATA_PATH} \
        --data_root ${DATA_ROOT} \
        --mask_type tube \
        --mask_ratio 0.3 \
        --input_size 160 \
        --mask_ratio_audio 0.3 \
        --input_size_audio 64 \
        --model pretrain_hicmae_dim512_patch16_160_a256 \
        --encoder_depth 10 \
        --decoder_depth 4 \
        --encoder_depth_audio 10 \
        --decoder_depth_audio 4 \
        --encoder_fusion_depth 2 \
        --batch_size 128 \
        --num_frames 5 \
        --sampling_rate 1 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 5 \
        --save_ckpt_freq 10 \
        --epochs 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --lr 3e-4 \
        --num_workers 8 \
        --roll_mag_aug True \
        --return_intermediate_features 3 6 9 \
        --loss_weight 0.0025 \
        --inter_contrastive_temperature 0.07

