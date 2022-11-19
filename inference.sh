PYTHON="/home2/jmeng15/anaconda3/envs/myenv/bin/python"

lambda=0.1

$PYTHON main_avgVth.py \
    --dataset ncars \
    --epochs 200 \
    --T 16 \
    --TET True \
    --lamb ${lambda} \
    --batch-size 32 \
    --lvth False \
    --neg -2.0 \
    --membit 2 \
    --resume "/home2/jmeng15/LV-Surrogate-Gradient-TE-SNN-VGG/save/NCARS/TET/MBNet/training_lambda0.1_PreSpikeAvgPool_Thre1.0_tau0.5_t16_4bit/checkpoint.pth.tar" \
    --save_path "/home2/jmeng15/LV-Surrogate-Gradient-TE-SNN-VGG/save/NCARS/TET/MBNet/training_lambda0.1_PreSpikeAvgPool_Thre1.0_tau0.5_t16_4bit/eval/" \
    --wbit 4 \
    --fine_tune \
    --evaluate 
    