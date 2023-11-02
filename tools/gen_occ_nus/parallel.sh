#!/bin/bash

# 进程数
MAX_PARALLEL=50

run_function() {
    local index=$1
    echo "Launch $index"
    sleep $((RANDOM % 10))
    python tools/gen_occ_nus/occ_label.py $index
}

for ((i = 0; i <= 850; i++)); do
    run_function $i &

    if (( $(jobs -p | wc -l) >= MAX_PARALLEL )); then
        wait -n
    fi
done

wait