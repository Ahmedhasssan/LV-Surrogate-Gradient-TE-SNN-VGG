PYTHON="/home/ahasssan/anaconda3/envs/myenv/bin/python"

lambda=0.1

$PYTHON main_avgVth.py \
    --dataset dvscifar10 \
    --epochs 200 \
    --T 30 \
    --TET True \
    --lamb ${lambda} \
    --batch-size 32 \
    --lvth False \
    --neg -1.0 \
    --membit 2 \
    --resume "/home/ahasssan/ahmed/LV-Surrogate-Gradient-TE-SNN-VGG/save/new/greater_equal_Vth/DVS_CIFAR10/TET/MBNetWide/training_lambda0.1_PreSpikeAvgPool_log2_fix_values_new_grid/checkpoint.pth.tar" \
    --save_path "./save/DVS_CIFAR10/TET/MBNet/training_PreSpikeAvgPool/eval/fixmap_alpha2" \
    --evaluate
    