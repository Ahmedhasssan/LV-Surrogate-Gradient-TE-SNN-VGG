PYTHON="/home2/jmeng15/anaconda3/envs/myenv/bin/python"

lambda=0.7

$PYTHON main_avgVth_cifar10.py \
    --epochs 200 \
    --T 20 \
    --TET True \
    --lamb ${lambda} \
    --batch-size 32 \
    --lvth True \
    --save_path "./save/cifar10/lvth/vgg7/training_lambda${lambda}_SkipFirstLayer/"
