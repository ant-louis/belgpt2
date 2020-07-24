#!/bin/bash
#----------------------------------------------------------------
# MODELS
#----------------------------------------------------------------
# GPT-2 small: 12-layer, 768-hidden, 12-heads, 117M parameters.
SIZE=small
MODEL=gpt2
BS=5

# GPT-2 medium: 24-layer, 1024-hidden, 16-heads, 345M parameters.
#SIZE=medium
#MODEL=gpt2-$SIZE
#BS=2

#----------------------------------------------------------------
# DATA FILES
#----------------------------------------------------------------
TRAIN_FILE=/raid/antoloui/THESIS/-/Projects/belgpt2/data/fr.train
EVAL_FILE=/raid/antoloui/THESIS/-/Projects/belgpt2/data/fr.dev

#----------------------------------------------------------------
# OUTPUT
#----------------------------------------------------------------
OUTPUT=/raid/antoloui/THESIS/-/Projects/belgpt2/models/gpt2/$SIZE
CACHE=/raid/antoloui/THESIS/-/Projects/belgpt2/cache

#----------------------------------------------------------------
# TRAINING PARAMETERS
#----------------------------------------------------------------
TOKENIZER=/raid/antoloui/THESIS/-/Projects/belgpt2/models/bpe/byte/
BLOCK_SIZE=1024

EPOCHS=10
WARMUP_RATIO=0.01
LR=1e-4

WEIGHT_DECAY=1e-2
ADAM_EPS=1e-6
FP16=O1


#----------------------------------------------------------------
# STEPS
#----------------------------------------------------------------
SAVING_STEPS=10000
MAX_CHECKPOINTS=10
LOGGING_STEPS=100



#----------------------------------------------------------------
# LAUNCHING TRAINING
#----------------------------------------------------------------
python -W ignore -u tools/run_language_modeling.py \
        --model_type gpt2 \
        --model_name_or_path $MODEL \
        --tokenizer_path $TOKENIZER \
        --output_dir $OUTPUT \
        --overwrite_output_dir \
        --do_train \
        --train_data_file $TRAIN_FILE \
        --per_gpu_train_batch_size $BS \
        --num_train_epochs $EPOCHS.0 \
        --block_size $BLOCK_SIZE \
        --learning_rate $LR \
        --weight_decay $WEIGHT_DECAY \
        --adam_epsilon $ADAM_EPS \
        --warmup_ratio $WARMUP_RATIO \
        --save_steps $SAVING_STEPS \
        --logging_steps $LOGGING_STEPS \
        --do_eval \
        --eval_data_file $EVAL_FILE \
        --save_total_limit $MAX_CHECKPOINTS \
        --fp16 \
        --fp16_opt_level $FP16 \
        --cache_dir $CACHE |& tee $OUTPUT/training_logs.txt