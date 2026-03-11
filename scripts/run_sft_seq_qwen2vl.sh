export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=./logs/seq/qwen2vl/test

export CUDA_VISIBLE_DEVICES=0,2,3

# srun accelerate launch --config_file=/mnt/petrelfs/yanziang/videoo1/TimeZero/configs/zero3.yaml 
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12951" \
    train/sft_seq.py \
    --deepspeed scripts/zero3_offload.json \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --train_data_path data/annotations/charades/train.json \
    --eval_data_path data/annotations/charades/charadea_sta_seq_test.json \
    --video_folder /your_data_path/charades/videos \
    --dataset_name xxx \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --max_seq_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --fp16 \
    --torch_dtype float16 \
    --logging_steps 5 \
    --eval_strategy no \
    --report_to tensorboard \
    --output_dir $OUTDIR \
    --save_steps 500 \
    --save_only_model true \
    --use_peft \
    --lora_target_modules all-linear \