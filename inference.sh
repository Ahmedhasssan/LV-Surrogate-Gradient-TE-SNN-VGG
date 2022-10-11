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
    --resume "./save/DVS_CIFAR10/TET/MBNet/training_lambda0.1/checkpoint.pth.tar" \
    --save_path "./save/DVS_CIFAR10/TET/MBNet/training_lambda0.1/eval/" \
    --evaluate
    