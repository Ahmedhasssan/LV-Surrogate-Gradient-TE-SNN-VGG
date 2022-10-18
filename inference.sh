PYTHON="/home2/jmeng15/anaconda3/envs/myenv/bin/python"

lambda=0.1

$PYTHON main_avgVth.py \
    --dataset dvscifar10 \
    --epochs 200 \
    --T 30 \
    --TET True \
    --lamb ${lambda} \
    --batch-size 32 \
    --lvth False \
    --neg -1.0 \
    --membit 2 \
    --resume "./save/DVS_CIFAR10/TET/QMBNet/training_lambda0.1_PreSpikeAvgPool_W4bit/checkpoint.pth.tar" \
    --save_path "./save/DVS_CIFAR10/TET/QMBNet/training_lambda0.1_PreSpikeAvgPool_W4bit/eval/" \
    --evaluate
    