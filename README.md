# LT-SNN: Spiking Neural Network with Learnable Threshold for Event-based Classification and Object Detection

SpQuant-SNN introduces an innovative quantization strategy for spiking neural networks (SNNs), enabling **ultra-low precision membrane potentials** combined with **sparse activations**. This approach significantly improves efficiency and unlocks the potential for **on-device spiking neural network applications**, such as energy-efficient edge AI.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contact](#contact)
- [Papers](#papers)

---

## Features
- **SNN with learnable threshold**: Spiking neural networks with layer-wise learnable threshold.
- **Scalability**: Demonstrates high performance across various SNN architectures and event/static datasets.
- **Surrogate Gradient**: used Sigmoid Surrogate Gradient for backward pass optimization

---

## Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.9
- CUDA >= 11.2 (Optional, for GPU acceleration)

### Install Required Packages
Clone the repository and install dependencies:

```bash
git clone https://github.com/Ahmedhasssan/LV-Surrogate-Gradient-TE-SNN-VGG.git
cd LV-Surrogate-Gradient-TE-SNN-VGG
pip install -r requirements.txt
```

## Usage

### Preprocess of DVS-CIFAR
 * Download CIFAR10-DVS dataset
 * Use dvsloader to make dataloader

## Files to Run
1. To initiate the training run "python3 main.py".
2. Run "python3 main_avgVth.py" to start training where 1 is replaced with Avg_learnable Threshold (Here temporal weightage is 0.9)

### Example Scripts
The repository includes examples for training and evaluating LT-SNNs on popular Event (**DVS-MNIST**, **DVS-CIFAR-10**) and Static image (**MNIST**, **CIFAR-10** and **Caltech-101**) datasets:

```bash
python3 main.py
python3 main_avgVth.py
```

## Methodology

LT-SNN introduces:

1. **Learnable threshold**:
   - Optimize layer-wise potential threshold to improve the firing operation.

2. **Layer-wise Dynamics**:
   - Captures layer-wise membrane potential dynamics across complex datasets.

3. **Sigmoid surrogate gradient**:
   - Fully optimize the backward pass in SNN by using gradient surrogation.

For a detailed explanation, refer to our [paper](https://ieeexplore.ieee.org/abstract/document/10650320).

## Results

SpQuant-SNN achieves state-of-the-art performance on spiking neural network benchmarks while dramatically reducing resource usage.

## Dataset: DVS-CIFAR10

| **Dataset**       | **Method**            | **SNN Architecture**       | **# of Parameters** | **Weight Precision** | **Simulation Length** | **Top-1 Accuracy** |
|--------------------|-------------------|----------------------------|---------------------|----------------------|-----------------------|--------------------|
| **DVS-CIFAR-10**   | **LT-SNN**        | VGG-11                     | 9.34M               | 32-bit               | 30                    | 79.51%             |
|                    | **LT-SNN**        | MobileNet-V1 (light)       | 1.28M               | 32-bit               | 30                    | 75.70%             |
|                    | **LT-SNN**        | VGG-7                      | 1.91M               | 32-bit               | 30                    | 80.20%             |
|                    | **LT-SNN**        | VGG-9                      | 7.07M               | 4-bit                | 30                    | 80.07%             |
|                    | **LT-SNN**        | VGG-9                      | 7.07M               | 32-bit               | 10                    | 79.10%             |
|                    | **LT-SNN**        | VGG-9                      | 7.07M               | 32-bit               | 8                     | 78.30%             |
|                    | **LT-SNN**        | Spikformer-16-256          | 4.15M               | 32-bit               | 10                    | 79.00%             |

## Dataset: CIFAR-10

| **Dataset**       | **Method**            | **SNN Architecture**       | **# of Parameters** | **Weight Precision** | **Simulation Length** | **Top-1 Accuracy** |
|--------------------|-------------------|----------------------------|---------------------|----------------------|-----------------------|--------------------|
| **CIFAR-10**       | **LT-SNN**        | ResNet-19 | 12.31M | 32-bit | 2 | 94.19% |
|                    | **LT-SNN**        | ResNet-19 | 12.31M | 32-bit | 6 | 94.56% |
|                    | **LT-SNN**        | Spikformer-4-256 | 4.15M  | 32-bit | 4 | 95.19%   |

## Dataset: Prophesee-Gen1 Automotive Datase

| **Method**         | **Model Architecture**  | **SNN** | **Threshold** | **mAP** |
|---------------------|-------------------------|---------|---------------|---------|
| **LT-SNN**       | Custom-YoloV2-SNN     | Yes     | Fixed         | 0.122   |
| **LT-SNN**       | Custom-YoloV2-SNN     | Yes     | Learnable     | 0.298   |


Experimental results of LT-SNN on DVS-CIFAR10 datasets using different simulation lengths. These results highlight the effectiveness of LT-SNN in achieving high accuracy and energy efficiency for edge AI applications.

## Contact

For any inquiries or collaboration opportunities, feel free to reach out:

- **Email**: [ah2288.@cornell.edu](mailto:ah2288@cornell.edu)
- **GitHub**: [Ahmedhasssan](https://github.com/Ahmedhasssan)

## Papers

Here are the papers related to this repository:

1. **LT-SNN**: Hasssan, A., Meng, J., & Seo, J. S. (2024, June). Spiking Neural Network with Learnable Threshold for Event-based Classification and Object Detection. In 2024 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.*. [Link to paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10650320).


We welcome feedback, suggestions, and contributions to enhance LT-SNN!
