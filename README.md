# LV-TE using Surrogate Gradient for VGG11 and VGG7

## Prerequisites
The Following Setup is tested and it is working:
 * Python>=3.5
 * Pytorch>=1.9.0
 * Cuda>=10.2

## Preprocess of DVS-CIFAR
 * Download CIFAR10-DVS dataset
 * Use dvsloader to make dataloader

## Description
 * used Sigmoid Surrogate Gradient

## Files to Run
1. To initiate the training run "python3 main.py". This will start the training using mean value for TET loss as 1 (Here temporal weightage is 0.45)
2. Run "python3 main_avgVth.py" to start training where 1 is replaced with Avg_learnable Threshold (Here temporal weightage is 0.9)
