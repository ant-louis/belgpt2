#! bin/bash
# Antoine Louis (antoiloui@gmail.com)
# Modified from
# https://github.com/getalp/Flaubert

set -e

# Specify parameters to run the script
DATA_PATH=$1 # path to where you save the file to be split
dname=`dirname $DATA_PATH`
fname=`basename $DATA_PATH`

# Create directories
mkdir -p $dname/split/train
mkdir -p $dname/split/dev
mkdir -p $dname/split/test

# Split into train / valid / test
echo "***** Split into train / validation / test datasets *****"

split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
};

    local train_percent=$2 # percent of data to split for train
    local val_percent=$3 # percent of data to split for validation (also percent for test)

    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=`echo "($train_percent * $NLINES)/1" | bc`;
    num_val=`echo "($val_percent * $NLINES)/1" | bc`;
    NVAL=$((NTRAIN + num_val));
    num_test=$((NLINES - NVAL));

    shuf --random-source=<(get_seeded_random 42) $1 | (head -$NTRAIN ; dd status=none of=/dev/null)                 > $4;
    shuf --random-source=<(get_seeded_random 42) $1 | (head -$NVAL ; dd status=none of=/dev/null) | tail -$num_val  > $5;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -$num_test                                               > $6;
}
split_data $DATA_PATH 0.99 0.005 $dname/split/train/$fname.train $dname/split/dev/$fname.dev $dname/split/test/$fname.test

echo "Done"