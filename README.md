# Conditional GAN for MNIST (Linear Architecture)

## Overview
This project implements a **Conditional Generative Adversarial Network (CGAN)** for the MNIST handwritten digits dataset, using a simple and effective **fully connected (linear) architecture** for both the generator and discriminator. The model is implemented in PyTorch Lightning for modularity and ease of training.

## Features
- **Conditional Generation**: Generate images of a specific digit (0–9) by conditioning both the generator and discriminator on the label.
- **Linear (Fully Connected) Architecture**: Both networks use only linear layers, making the code simple, fast, and easy to understand.
- **One-hot Label Conditioning**: Labels are encoded as one-hot vectors and concatenated to the noise vector (generator) and image vector (discriminator).
- **Stable Training**: Uses batch normalization, LeakyReLU activations, and the Adam optimizer for robust and stable convergence.
- **PyTorch Lightning**: Modular code structure for easy experimentation and reproducibility.
- **Quick Training**: Trains in minutes on CPU or GPU; 5–20 epochs is usually sufficient for good results.

## Why Linear CGAN for MNIST?
- **Simplicity**: Minimal code, easy to debug and extend. Ideal for learning and research.
- **Speed**: Trains extremely fast due to low parameter count and small image size (28x28).
- **Stability**: Less prone to mode collapse and instability than deeper or convolutional GANs on MNIST.
- **Sufficient Capacity**: For MNIST, linear models are sufficient to generate high-quality, label-conditional samples. More complex architectures do not yield significant improvements.
- **Reproducibility**: Results are highly reproducible and less sensitive to hyperparameters.

## Empirical Results
- **Loss Curves**: Discriminator and generator losses converge smoothly.
- **Sample Quality**: Generated digits are visually convincing for all classes.
- **Advanced Tricks**: Techniques like spectral norm, self-attention, or deep CNNs do not significantly improve results on MNIST compared to this linear CGAN.

## Code Structure
- `model.py`: Contains the Generator, Discriminator, and GAN LightningModule.
- `data.py`: Loads and preprocesses the MNIST dataset using PyTorch Lightning DataModule.
- `train.py`: Training script (5 epochs by default).

## Usage
1. **Install dependencies** (PyTorch, torchvision, pytorch-lightning).
2. **Run training**:
   ```bash
   python train.py
   ```
3. **Monitor results**: Generated images and loss curves are logged for each epoch.

## References
- [Mirza & Osindero, 2014: Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## When to Use This Model
- **Educational purposes**: Learn GANs and conditional generation with minimal code.
- **Research baseline**: Use as a baseline for MNIST or other simple datasets.
- **Quick prototyping**: Rapidly test new ideas in a stable, reproducible setting.
