#!/bin/sh

export OUT_DIR=/raid/antoloui/THESIS/-/Projects/belgpt2/models/belgpt2-small/
export DEV_FILE=/raid/antoloui/THESIS/-/Projects/belgpt2/data/fr.test #fr.test, fr.dev
export CACHE=/raid/antoloui/THESIS/-/Projects/belgpt2/cache/
export MODEL=/raid/antoloui/THESIS/-/Projects/belgpt2/models/belgpt2-small/checkpoint-3150000

python -W ignore -u tools/run_language_modeling.py \
    --model_type=gpt2 \
    --model_name_or_path=$MODEL \
    --do_eval \
    --eval_data_file=$DEV_FILE \
    --output_dir=$OUT_DIR \
    --cache_dir=$CACHE \
    --block_size=1024 \
    --per_gpu_eval_batch_size=5 \
    --eval_all_checkpoints