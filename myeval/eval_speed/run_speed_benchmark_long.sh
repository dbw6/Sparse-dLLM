#!/bin/bash
# Run speed benchmark for 2048/4096 length with/without prefix eviction

# Generate questions if not exist
python3 extract_gsm8k.py

MODEL_PATH='GSAI-ML/LLaDA-1.5'
DATA_PATH='gsm8k_questions.txt'
OUTPUT_DIR='results_speed_benchmark'

# 1. 2048, Eviction Enabled (Sparse Prefix)
echo "Running 2048, Eviction Enabled (Sparse Prefix)..."
python3 opencompass/myeval/eval_speed/llada_sparse_dllm.py \
    --model_path ${MODEL_PATH} \
    --model_type "llada_1.5_2048_evict" \
    --data_path ${DATA_PATH} \
    --data_type "gsm8k" \
    --output_dir ${OUTPUT_DIR} \
    --kernel_size 3 \
    --keep_ratio 0.5 \
    --gen_length 2048 \
    --steps 2048

# 2. 2048, Eviction Disabled (Dense Prefix)
echo "Running 2048, Eviction Disabled (Dense Prefix)..."
python3 opencompass/myeval/eval_speed/llada_sparse_dllm.py \
    --model_path ${MODEL_PATH} \
    --model_type "llada_1.5_2048_no_evict" \
    --data_path ${DATA_PATH} \
    --data_type "gsm8k" \
    --output_dir ${OUTPUT_DIR} \
    --kernel_size 3 \
    --keep_ratio 0.5 \
    --gen_length 2048 \
    --steps 2048 \
    --disable_prefix_cache_eviction

# 3. 4096, Eviction Enabled (Sparse Prefix)
echo "Running 4096, Eviction Enabled (Sparse Prefix)..."
python3 opencompass/myeval/eval_speed/llada_sparse_dllm.py \
    --model_path ${MODEL_PATH} \
    --model_type "llada_1.5_4096_evict" \
    --data_path ${DATA_PATH} \
    --data_type "gsm8k" \
    --output_dir ${OUTPUT_DIR} \
    --kernel_size 3 \
    --keep_ratio 0.5 \
    --gen_length 4096 \
    --steps 4096

# 4. 4096, Eviction Disabled (Dense Prefix)
echo "Running 4096, Eviction Disabled (Dense Prefix)..."
python3 opencompass/myeval/eval_speed/llada_sparse_dllm.py \
    --model_path ${MODEL_PATH} \
    --model_type "llada_1.5_4096_no_evict" \
    --data_path ${DATA_PATH} \
    --data_type "gsm8k" \
    --output_dir ${OUTPUT_DIR} \
    --kernel_size 3 \
    --keep_ratio 0.5 \
    --gen_length 4096 \
    --steps 4096 \
    --disable_prefix_cache_eviction

