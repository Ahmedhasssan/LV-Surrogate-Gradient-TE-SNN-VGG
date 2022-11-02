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
    --membit 2 \
    --resume "/home2/jmeng15/LV-Surrogate-Gradient-TE-SNN-VGG/save/Mobilenet/low_precision/checkpoint.pth.tar" \
    --save_path "/home2/jmeng15/LV-Surrogate-Gradient-TE-SNN-VGG/save/Mobilenet/low_precision/eval/" \
    --evaluate
    