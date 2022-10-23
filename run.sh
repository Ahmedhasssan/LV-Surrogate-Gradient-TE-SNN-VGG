PYTHON="/home2/ahasssan/anaconda3/envs/myenv/bin/python"

lambda=0.1

$PYTHON main_avgVth.py \
    --dataset dvscifar10 \
    --epochs 200 \
    --T 30 \
    --lr 5e-4 \
    --TET True \
    --lamb ${lambda} \
    --batch-size 32 \
    --lvth False \
    --save_path "./save/DVS_CIFAR10/TET/MBNetWide/training_lambda${lambda}_PreSpikeAvgPool_NegQ/"
    