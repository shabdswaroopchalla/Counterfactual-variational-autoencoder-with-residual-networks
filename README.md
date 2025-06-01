# Counterfactual Variational Autoencoder with Residual Networks (ResNet-CI-VAE)

Welcome to the official repository for **ResNet-CI-VAE**, a major M.Tech project exploring *causal representation learning* through the integration of **counterfactual interventions**, **Residual Networks (ResNet)**, and **Structural Causal Models (SCMs)**. This project builds upon the CFI-VAE architecture and enhances it with learned partitioning and fusion modules using **Multilayer Perceptrons (MLPs)**.

---

##  Project Overview

The primary goal is to generate **causally disentangled representations** in the latent space of a Variational Autoencoder (VAE), which can perform robust counterfactual generation while maintaining high visual fidelity.

> Dataset Used: **CelebA** (focused on the "Beard" attribute subset)

---

##  Key Concepts

- **Disentanglement**: Separate latent space into causal and residual subspaces.
- **SCM Integration**: Learn causal structure via Structural Causal Models.
- **Counterfactual Inference**: Apply do-interventions on latent factors to simulate "what-if" scenarios.
- **Residual Learning**: Use ResNet-based encoder-decoder blocks for improved expressivity.
- **MLP-Based Partitioning & Fusion**: Learn to separate and recombine causal & non-causal components.

---

##  Architecture Components

1. **Encoder**  
   - Strategy 1: Standard Convolutional Encoder  
   - Strategy 2: ResNet-Based Encoder

2. **Partitioner (MLP)**  
   - Splits latent `z` into `z_causal` and `z_residual`

3. **SCM Layer**  
   - Enforces causal structure using a learned adjacency matrix

4. **Fusion Module (MLP)**  
   - Merges causal and residual vectors post-intervention

5. **Decoder**  
   - Strategy 1: Convolutional Decoder  
   - Strategy 2: ResNet-Based Decoder

6. **Causal Classifier**  
   - Computes Total Direct Effect (TDE) for causal interpretability

---

##  Dataset & Preprocessing

- **Dataset**: CelebA "Beard" attribute subset
- **Image Sizes**:
  - Strategy 1: `64x64`
  - Strategy 2: `128x128`
- **Steps**:
  - Attribute selection
  - Normalization
  - Train-test split

---

##  Training Configuration

- **Optimizer**: Adam
- **Learning Rate**: `1e-5`
- **Annealing Strategy**: Gradual KL divergence increase
- **Regularization**: Early stopping, latent norm penalties
- **Logging**: Checkpointing and visualization integrated
- **Frameworks Used**: PyTorch, NumPy, Matplotlib

