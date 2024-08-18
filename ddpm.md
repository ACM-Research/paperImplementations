# DDPM (Denoising Diffusion Probabilistic Models)

## Key Concept

Iterative Denoising Process:
DDPM gradually transforms noise into a desired image through a series of denoising steps. This process allows for high-quality image generation by learning to reverse a fixed Markov chain that gradually adds noise to data.

## Strengths

- High-quality image generation
  > DDPMs can generate samples of comparable quality to GANs on image synthesis benchmarks, while offering a more stable training process.
- Flexible architecture
  > The model can be adapted to various image sizes and types without significant changes to the core architecture.
- Tractable likelihood computation
  > Unlike GANs, DDPMs allow for direct optimization of log-likelihood, providing a clear training objective.

## Weaknesses

- Slow sampling process
  > DDPMs typically require many iterations to generate a single image, which can be computationally expensive and time-consuming.
- Large memory requirements
  > The model needs to store intermediate states during the reverse process, which can lead to high memory usage.
- Potential for mode collapse
  > While less prone than GANs, DDPMs can still suffer from mode collapse in some scenarios.

## Novelties

- Introducing a diffusion-based generative model for images
- Connecting the diffusion process to denoising score matching and Langevin dynamics
- Demonstrating competitive performance with state-of-the-art GANs

# Summary

Core Idea
The paper introduces a new class of generative models that use a diffusion process to transform a simple distribution (like Gaussian noise) into a complex data distribution (such as images).

Methodology
Diffusion Process: The model learns to reverse a diffusion process. It starts from a data sample and progressively adds noise over multiple steps, eventually leading to pure noise. The model then learns to reverse this process to recover the original data from noise.

Training Objective: The training involves learning a neural network that estimates the noise added at each diffusion step. The objective is to minimize the difference between the estimated and actual noise, which helps in generating high-quality samples from the noise.

Noise Schedule: The model uses a fixed noise schedule for adding noise during training. The reverse process is learned through a neural network that models the distribution of the data conditioned on the noisy data.

## Previous Literature

- Score-based generative models and noise-conditioned score networks (NCSN) laid groundwork for gradient-based sampling methods
- Variational autoencoders (VAEs) introduced the concept of latent variable models for image generation

## Comparison

- Unlike GANs, DDPMs don't rely on adversarial training, potentially leading to more stable optimization
- Compared to VAEs, DDPMs offer better sample quality and more flexible latent space

## Model Architecture

- Uses a U-Net architecture with skip connections
- Incorporates time embeddings to condition the model on the noise level
- Employs self-attention layers for global context

## Training Process

- Forward process: Gradually add Gaussian noise to images
- Reverse process: Learn to denoise images step by step
- Loss function: Combination of variational lower bound and simplified objective

## Sampling

- Start from pure noise and iteratively apply the learned denoising process
- Can use fewer sampling steps than training steps for faster generation

## Image Generation Benchmarks

- CIFAR-10, CelebA-HQ, LSUN
- Evaluation metrics: Inception Score (IS), Fréchet Inception Distance (FID)

## Ablation Studies

- Impact of noise schedule
- Effect of model depth and width
- Importance of self-attention layers

## Likelihood Estimation

- Comparison with other likelihood-based models (e.g., PixelCNN++, VQ-VAE)

# Conclusion

- DDPMs offer a promising new approach to generative modeling
- Competitive performance with GANs while providing tractable likelihoods
- Future work: Improving sampling speed, exploring applications beyond image generation
