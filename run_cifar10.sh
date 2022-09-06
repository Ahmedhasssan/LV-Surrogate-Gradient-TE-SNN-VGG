PYTHON="/home2/jmeng15/anaconda3/envs/myenv/bin/python"

lambda=0.7

$PYTHON main_avgVth_cifar10.py \
    --epochs 200 \
    --T 6 \
    --TET True \
    --lr 0.1 \
    --lamb ${lambda} \
    --batch-size 128 \
    --lvth False \
    --save_path "./save/cifar10/TETBaseline/resnet19/training_lambda${lambda}_SkipFirstLayer_SGD/"
