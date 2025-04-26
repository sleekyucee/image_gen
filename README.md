#Deep Generative Models for Facial Image Synthesis

This repository presents a comparative study of deep generative models for facial image synthesis, using the FFHQ dataset. Three architectures were implemented and evaluated: Variational Autoencoder (VAE), Deep Convolutional GAN (DCGAN), and StyleGAN2.

#Projects Included:

- VAE Project (vae_project/): Variational Autoencoder for face image reconstruction and generation.

- DCGAN Project (dcgan_project/): Deep Convolutional GAN for face image synthesis.

- StyleGAN2 Project (stylegan2_project/): StyleGAN2-based face generator with advanced style-based architecture.

#Dataset:

Flickr-Faces-HQ (FFHQ) dataset (~70,000 high-quality face images).

Images were resized to smaller resolutions (e.g., 64×64, 128×128) for model compatibility and faster training.

#Model Overview:

- VAE: Encoder-decoder architecture with latent space sampling under Gaussian assumptions.

- DCGAN: Convolutional GAN framework utilizing transposed convolutions, batch normalization, and LeakyReLU activations.

- StyleGAN2: Style-based generator with progressive growing, style mixing regularization, and path length regularization.

#Training:

- Model training was performed primarily on the Hyperion GPU cluster at City, University of London.

- Google Colab Pro (A100 GPUs) was used for exploratory data analysis (EDA), inferencing, and final FID evaluation.

- Training logs, intermediate samples, and hyperparameter tuning were tracked using Weights & Biases (wandb).

#Evaluation:

- Quantitative evaluation using Fréchet Inception Distance (FID) to assess sample quality.

- Qualitative evaluation by visual inspection of generated samples across models.

#How to Run: 
Each project (vae_project/, dcgan_project/, stylegan2_project/) contains its own train.py and configuration files under configs/.

#Example (training the VAE):

cd vae_project python train.py --config configs/config_ld128_beta1.yaml

#Requirements: 
Install all dependencies using:

pip install -r requirements.txt

#Acknowledgements:

FFHQ Dataset by NVIDIA.

StyleGAN2 paper and open-source PyTorch implementations.

This project was completed as part of a deep learning coursework project on generative models. All models were trained, evaluated, and analyzed successfully.