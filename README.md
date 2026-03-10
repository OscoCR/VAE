<!--             
<style>
  .texttt {
    font-family: Consolas; /* Monospace font */
    font-size: 1em; /* Match surrounding text size */
    color: teal; /* Text color */
    letter-spacing: 0; /* Adjust if needed */
  }
</style> -->

<h1 align="center">
  <span style="color: teal; font-family: Consolas;">Learning Compact Representations of Agricultural Fields</span>: A Study of Variational Autoencoders Variants for Aerial Drone Imagery
</h1>

<div align="center">
  <a href="mailto:dxie@ic-itcr.ac.cr" target="_blank">Danny&nbsp;Xie-Li</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="mailto:gonzalezhernandez.manfred@gmail.com" target="_blank">Manfred&nbsp;Gonzalez-Hernandez</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="mailto:blaz.meden@fri.uni-lj.si" target="_blank">Blaz&nbsp;Meden</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="mailto:fabian.fallasmoya@ucr.ac.cr" target="_blank">Fabian&nbsp;Fallas-Moya</a><sup>3</sup>
  <br>
  <sup>1</sup> Instituto Tecnológico de Costa Rica, Cartago, Costa Rica &emsp; <br>
  <sup>2</sup> Computer Vision Lab, University of Ljubljana, Ljubljana, Slovenia &emsp; <br>
  <sup>3</sup> Sede del Atlántico, Imagine Lab, Universidad de Costa Rica, Cartago, Costa Rica &emsp;
</div>

---
<p align="center">
  <img src="assets/Pipeline_Beta_VQ.jpg?raw=true" width="99.1%" />
</p>

## 📝 Abstract

The integration of aerial drone imagery into precision agriculture enables large-scale, high-resolution monitoring of crop conditions but also introduces challenges due to the dimensionality and variability of visual data. Variational Autoencoders (VAEs) and their variants provide a promising framework for learning compact latent representations that preserve meaningful crop features while reducing computational complexity. In this work, we investigate the suitability of VAE-based architectures for analyzing aerial imagery of pineapple fields. Using a dataset of approximately $5000$ drone-acquired images, we evaluate reconstruction fidelity, robustness to noise, and the structure of latent spaces across different VAE variants, including VQ-VAE and $\beta$-VAE. Our analysis demonstrates that VAEs not only capture fine-grained plant features but also enable latent-space clustering that separates foreground (pineapple plants) from background (soil, vegetation, and other elements). These findings highlight the potential of VAE-derived representations for supporting downstream tasks such as plant counting, yield estimation, and stress detection. By releasing both dataset and code at [https://github.com/imagine-laboratory/VAE](https://github.com/imagine-laboratory/VAE), we aim to establish a reproducible benchmark for generative models in agricultural monitoring, advancing data-driven approaches for sustainable pineapple cultivation.

## 📝 Overview

This repository provides implementations of Autoencoders and Variational Autoencoder (VAE) variants designed for learning compact, informative representations from aerial drone imagery of agricultural fields. These models are intended to support key tasks in **precision agriculture**, including **crop monitoring**, **yield estimation**, and **field analysis**.

## Installation

We tested our code using CUDA 12.8 and Python 3.10. To install requirements, run in the terminal:

```bash
pip3 install -r requirements.txt
```

## 🔍 Overview of Models

* **Autoencoder (AE)**
  A standard encoder–decoder model for unsupervised feature extraction and image reconstruction.

* **Variational Autoencoder (VAE)**
  A probabilistic generative model that learns a latent distribution over the input space, enabling stochastic sampling and smooth interpolation.

* **β-VAE**
  A variant of VAE that introduces a hyperparameter β to promote disentangled latent representations, useful for more interpretable factors of variation.

* **Vector-Quantized VAE (VQ-VAE)**
  A discrete latent variable model that uses vector quantization to learn a finite set of latent embeddings, improving interpretability and compression.

## 🚀 Getting Started

To train or evaluate a model, use the following command:

```bash
python main.py --config ./configs/vqvae.yaml --model vqvae
```

## 📚 Citation

If you use this repository or the dataset in your research, please cite:

```bibtex
@article{xie2025learningVAEagriculture,
  title={Learning Compact Representations of Agricultural Fields: A Study of Variational Autoencoder Variants for Aerial Drone Imagery},
  author={Xie-Li, Danny and Gonzalez-Hernandez, Manfred and Meden, Blaz and Fallas-Moya, Fabian},
  year={2025},
  url={https://github.com/imagine-laboratory/VAE}
}
````




