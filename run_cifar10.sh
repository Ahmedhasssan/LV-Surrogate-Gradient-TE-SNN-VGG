PYTHON="/home2/jmeng15/anaconda3/envs/myenv/bin/python"

lambda=0.7

$PYTHON main_avgVth_cifar10.py \
    --epochs 200 \
    --T 6 \
    --TET True \
    --lamb ${lambda} \
    --batch-size 32 \
    --lvth True \
    --save_path "./save/cifar10/TETBaseline/resnet19/training_lambda${lambda}_SkipFirstLayer/"
