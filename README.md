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

| **Method**                   | **Architecture**       | **Weight Precision** | **Umem Precision** | **Weight Memory (MB)** | **Umem Memory (MB)** | **Total Memory (MB)** | **FLOPs Reduction** | **Top-1 Accuracy**       |
|------------------------------|------------------------|-----------------------|--------------------|-------------------------|-----------------------|-----------------------|--------------------|--------------------------|
| Our work (SNN-BL)            | VGG-9                 | 32-bit               | 32-bit             | 41.12                  | 3.68                 | 48.58                | 1×                | 78.45%                  |
| Our work (Quant-SNN)         | VGG-9                 | 2-bit                | 1.58-bit           | 2.57                   | 0.23                 | 3.75                 | 1×                | 77.94% (-0.51)          |
| Our work (SpQuant-SNN)       | VGG-9                 | 2-bit                | 1.58-bit           | 2.57                   | 0.23                 | 3.75                 | 5.0×              | 76.80% (-1.14)          |              |

## Dataset: CIFAR10

| **Method**                   | **Architecture**       | **Weight Precision** | **Umem Precision** | **Weight Memory (MB)** | **Umem Memory (MB)** | **Total Memory (MB)** | **FLOPs Reduction** | **Top-1 Accuracy**       |
|------------------------------|------------------------|-----------------------|--------------------|-------------------------|-----------------------|-----------------------|--------------------|--------------------------|
| Our work (SNN-BL)  | ResNet-19 | 32-bit | 32-bit | 49.94 | 5.5 | 60.94 | 1× | 94.56%|
| Our work (Quant-SNN) | ResNet-19 | 4-bit | 1.58-bit | 6.24 | 0.25 | 7.49 | 1× | 94.11% (-0.45)|
| Our work (Quant-SNN) | Spikformer-4-256 | 8-bit | 1.58-bit | 9.62 | 0.25 | 15.26 | 1× | 94.99% (-0.52)|
| Our work (SpQuant-SNN) | ResNet-19 | 4-bit | 1.58-bit | 6.24 | 0.25 | 7.49 | 5.1× | 93.09% (-1.48)|


Experimental results of Quant-SNN and SpQuant-SNN on DVS datasets using T = 10. These results highlight the effectiveness of SpQuant-SNN in achieving high accuracy and energy efficiency for edge AI applications.

## Contact

For any inquiries or collaboration opportunities, feel free to reach out:

- **Email**: [ah2288.@cornell.edu](mailto:ah2288@cornell.edu)
- **GitHub**: [Ahmedhasssan](https://github.com/Ahmedhasssan)

## Papers

Here are the papers related to this repository:

1. **IM-SNN**: Hasssan, A., Meng, J., Anupreetham, A., & Seo, J. S. (2024, August). IM-SNN: Memory-Efficient Spiking Neural Network with Low-Precision Membrane Potentials and Weights. IEEE/ACM International Conference on Neuromorphic Systems (ICONS).*. [Link to paper](https://par.nsf.gov/biblio/10545833).
2. **Sp-QuantSNN**: Hasssan, Ahmed, Jian Meng, Anupreetham Anupreetham, and Jae-sun Seo. "SpQuant-SNN: ultra-low precision membrane potential with sparse activations unlock the potential of on-device spiking neural networks applications." Frontiers in Neuroscience 18 (2024): 1440000. [Link to paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1440000/full).


We welcome feedback, suggestions, and contributions to enhance SpQuant-SNN!
