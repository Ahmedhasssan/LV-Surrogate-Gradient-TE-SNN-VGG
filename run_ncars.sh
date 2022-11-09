PYTHON="/home2/jmeng15/anaconda3/envs/myenv/bin/python"

lambda=0.1

$PYTHON main_avgVth.py \
    --dataset ncars \
    --epochs 200 \
    --T 16 \
    --lr 5e-4 \
    --TET True \
    --lamb ${lambda} \
    --batch-size 32 \
    --lvth False \
    --thres 1.0 \
    --tau 0.5 \
    --wbit 32 \
    --save_path "./save/NCARS/TET/MBNet/training_lambda${lambda}_PreSpikeAvgPool_Thre1.0_tau0.5_t16/"
    