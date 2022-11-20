PYTHON="/home/jmeng15/anaconda3/bin/python"

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
    --resume "/home/jmeng15/LV-Surrogate-Gradient-TE-SNN-VGG/save/TapeOut112522/checkpoint.pth.tar" \
    --save_path "/home/jmeng15/LV-Surrogate-Gradient-TE-SNN-VGG/save/TapeOut112522/eval/" \
    --evaluate
    