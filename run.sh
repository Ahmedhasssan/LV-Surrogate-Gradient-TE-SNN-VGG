PYTHON="/home2/jmeng15/anaconda3/envs/myenv/bin/python"

lambda=0.05

$PYTHON main_avgVth.py \
    --dataset dvscifar10 \
    --epochs 200 \
    --model resnet19 \
    --T 10 \
    --lr 1e-3 \
    --TET True \
    --lamb ${lambda} \
    --batch-size 32 \
    --lvth True \
    --save_path "./Rebuttal/resnet18/200epochs/ltsnn/"
    