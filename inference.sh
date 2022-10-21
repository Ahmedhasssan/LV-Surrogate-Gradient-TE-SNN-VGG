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
    --neg -2.0 \
    --membit 4 \
    --thres 1.0 \
    --tau 0.5 \
    --wbit 32 \
    --resume "./save/DVS_CIFAR10/TET/MBNet/training_lambda0.1_PreSpikeAvgPool_Thre1.0_tau0.5/checkpoint.pth.tar" \
    --save_path "./save/DVS_CIFAR10/TET/MBNet/training_lambda0.1_PreSpikeAvgPool_Thre1.0_tau0.5/eval/" \
    --evaluate
    