#!/bin/bash
# Antoine Louis (antoiloui@gmail.com)

FILES=$1  #/raid/antoloui/THESIS/-/Projects/CloneAI/Data/French_corpus/fr.train
METHOD=$2  #byte
VOCAB_SIZE=$3  #50257
OUTPUT=$4/$METHOD  #/raid/antoloui/THESIS/-/Projects/CloneAI/Code/_models/bpe


python tools/learn_bpe.py \
    --files $FILES \
    --method $METHOD \
    --vocab_size $VOCAB_SIZE \
    --outdir $OUTPUT