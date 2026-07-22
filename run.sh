#!/bin/bash
#SBATCH --job-name=iTransformer
#SBATCH --partition=h100base-8
#SBATCH --nodelist=wf-a3-megagpu-8g-base-2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_%j.log

source activate itrans
cd /home/jli038/code/SC/iTransformer
model_name=iTransformer


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/jlab/ \
  --data_path jlab.csv \
  --model_id jlab_48_48 \
  --model $model_name \
  --data jlab \
  --features MS \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 48  \
  --batch_size 32 \
  --e_layers 2 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 3 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --train_epochs 20 \
  --itr 1