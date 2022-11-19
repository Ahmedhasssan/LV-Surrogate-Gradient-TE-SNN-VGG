PYTHON="/home/ahasssan/anaconda3/envs/myenv/bin/python"

lambda=0.45

$PYTHON main_avgVth.py \
    --dataset dvscifar10 \
    --epochs 200 \
    --model VGGSNN9 \
    --T 30 \
    --lr 5e-4 \
    --TET True \
    --lamb ${lambda} \
    --batch-size 32 \
    --lvth True \
    --save_path "./Rebuttal/VGG9SNN/200epochs/"
    