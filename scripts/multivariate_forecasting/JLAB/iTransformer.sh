export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/jlab/ \
  --data_path jlab.csv \
  --model_id jlab_48_48 \
  --model $model_name \
  --data jlab \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96  \
  --batch_size 16 \
  --e_layers 2 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 3 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --train_epochs 20 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 16 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 128 \
  --train_epochs 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1