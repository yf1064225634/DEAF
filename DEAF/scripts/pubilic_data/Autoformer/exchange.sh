#日志保存位置
if [ ! -d "./logs/public/Autoformer" ]; then
    mkdir ./logs/public/Autoformer
fi
#已知序列长度；预测序列长度
seq_len=96
label_len=64
dim=8 #维度
for pred_len in 64 96 128
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/EX/ \
    --data_path exchange_rate.csv \
    --model_id exchange_$pred_len \
    --model Autoformer \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in $dim \
    --dec_in $dim \
    --c_out $dim \
    --itr 1 \
    --train_epochs 100 >logs/public/Autoformer/'exchange_'$pred_len.log
done
