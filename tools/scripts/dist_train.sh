#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

# Set NCCL environment variables for error handling
#export NCCL_BLOCKING_WAIT=1
#export NCCL_ASYNC_ERROR_HANDLING=1

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT


python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS} --max_ckpt_save_num 20 --num_epochs_to_eval 5


