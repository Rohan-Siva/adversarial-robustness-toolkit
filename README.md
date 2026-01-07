# Adversarial Robustness Toolkit

## Overview
The Adversarial Robustness Toolkit is a comprehensive PyTorch-based framework designed for researching and evaluating the safety of machine learning models. It provides implementations of state-of-the-art adversarial attacks and defense mechanisms, allowing researchers and developers to test model robustness against malicious perturbations.

## Tools Used
- **Deep Learning**: PyTorch, Torchvision
- **Data Manipulation**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Utilities**: Scikit-learn, Tqdm

## Structure
- `attacks/`: Contains implementations of adversarial attacks (e.g., FGSM, PGD, C&W).
- `defenses/`: Contains defense mechanisms such as adversarial training and defensive distillation.
- `models/`: Defines neural network architectures (e.g., ResNet, VGG) used for experiments.
- `evaluation/`: Scripts and tools for evaluating model performance under attack.
- `experiments/`: Scripts for running comprehensive robustness experiments.
- `config.py`: Central configuration file for setting hyperparameters, dataset paths, and attack settings.

## Setup

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended)

### Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
Edit `config.py` to adjust parameters such as:
- **Dataset**: CIFAR10, MNIST, or ImageNet.
- **Model**: ResNet18, ResNet50, VGG16, etc.
- **Attack Parameters**: Epsilon, step size, iterations.
- **Defense Settings**: Adversarial training ratio, distillation temperature.

## How It Works
1. **Configure**: Set your desired experiment parameters in `config.py`.
2. **Train/Load**: Use the provided scripts to train a model or load a pre-trained one.
3. **Attack**: Run attack scripts to generate adversarial examples.
4. **Defend**: Apply defense mechanisms to improve model robustness.
5. **Evaluate**: Measure the model's accuracy on both clean and adversarial data to assess robustness.

## Contact
For collaborations or questions, please reach out to rohansiva123@gmail.com.
