#!/bin/bash
# Antoine Louis (antoiloui@gmail.com)

lg=fr
DIRPATH=$1  #  /raid/antoloui/Projects/CloneAI/Data/French_corpus/processed/split/dev/
EXT=`basename $DIRPATH`  # dev, test, train
OUTFILE=$lg.$EXT

python tools/merge_files.py \
    --dirpath $DIRPATH \
    --ext $EXT \
    --outfile $OUTFILE