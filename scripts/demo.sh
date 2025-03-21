CONFIG=$1
CKPT=$2
FOLDER=${3:-"examples"}
SAVEDFOLDER=${4:-"exp/examples_demo"}
PORT=${5:-23411}

HOST=$(hostname -i)

python ./scripts/demo.py \
    --batch 1 \
    --gpus 0 \
    --world-size 1 \
    --flip-test \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --folder_name ${FOLDER} \
    --work_dir ${SAVEDFOLDER} \
