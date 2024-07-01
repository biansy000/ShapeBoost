CONFIG=$1
CKPT=$2
PORT=${3:-23411}
EXPID='val'

HOST=$(hostname -i)

python ./scripts/validate_smpl_shape_ddp.py \
    --nThreads 6 \
    --launcher pytorch --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} --seed 123123
