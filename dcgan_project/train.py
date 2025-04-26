import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#import helper functions and classes
from utils import load_config, dcgan_discriminator_loss, dcgan_generator_loss
from dataset import FFHQDataset
from model import Generator, Discriminator, weights_init_normal
from logger import Logger

#device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, flush=True)

#load configuration
if len(sys.argv) != 2:
    print("Usage: python train.py <config_path>", flush=True)
    sys.exit(1)
config_path = sys.argv[1]
config = load_config(config_path)

#determine dataset directory
if config["data_settings"].get("use_hyperion", False):
    dataset_root = config["data_settings"]["hyperion_dir"]
else:
    dataset_root = config["data_settings"]["root_dir"]

#prepare logger
experiment_name = config["experiment_settings"]["experiment_name"]
logger = Logger(experiment_name=experiment_name)

#create dataset and dataLoader
image_size = config["data_settings"]["image_size"]
batch_size = config["train_settings"]["batch_size"]
num_workers = config["train_settings"].get("num_workers", 4)

dataset = FFHQDataset(root_dir=dataset_root, image_size=image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

#create model instances
latent_dim = config["model_settings"]["latent_vector_size"]
img_channels = config["model_settings"].get("img_channels", 3)
gen_feature_maps = config["model_settings"].get("feature_maps_size_gen", 64)
dis_feature_maps = config["model_settings"].get("feature_maps_size_dis", 64)

generator = Generator(latent_dim=latent_dim, img_channels=img_channels, feature_maps=gen_feature_maps).to(device)
discriminator = Discriminator(img_channels=img_channels, feature_maps=dis_feature_maps).to(device)

#initialize model weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

#use DataParallel if multiple GPUs available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!", flush=True)
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

#set up optimizers and loss
lr = config["train_settings"]["lr"]
beta1 = config["train_settings"]["beta_optim"]

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

criterion = nn.BCELoss()

#labels
real_label_value = 1.0
fake_label_value = 0.0

#training settings
num_epochs = config["train_settings"]["nr_epochs"]

#initialize best losses
best_loss_D = float("inf")
best_loss_G = float("inf")

#prepare checkpoints and images directory
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("images", exist_ok=True)

print("Starting Training Loop...", flush=True)
for epoch in range(num_epochs):
    print(f"\n=== Starting Epoch {epoch+1}/{num_epochs} ===", flush=True)
    epoch_loss_D = 0.0
    epoch_loss_G = 0.0

    for i, imgs in enumerate(dataloader):
        b_size = imgs.size(0)
        real_imgs = imgs.to(device)

        #update discriminator
        discriminator.zero_grad()

        #real batch
        label_real = torch.full((b_size,), real_label_value, device=device)
        output_real = discriminator(real_imgs).view(-1)
        loss_D_real = criterion(output_real, label_real)

        #fake batch
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_imgs = generator(noise)
        label_fake = torch.full((b_size,), fake_label_value, device=device)
        output_fake = discriminator(fake_imgs.detach()).view(-1)
        loss_D_fake = criterion(output_fake, label_fake)

        #combine real and fake losses
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()

        #update Generator
        generator.zero_grad()
        label_real_for_G = torch.full((b_size,), real_label_value, device=device)
        output_fake_for_G = discriminator(fake_imgs).view(-1)
        loss_G = criterion(output_fake_for_G, label_real_for_G)
        loss_G.backward()
        optimizer_G.step()

        epoch_loss_D += loss_D.item()
        epoch_loss_G += loss_G.item()

        if i % 50 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(dataloader)}] "
                  f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} "
                  f"D(x): {output_real.mean().item():.4f} D(G(z)): {output_fake.mean().item():.4f}", flush=True)

    #end of epoch
    avg_loss_D = epoch_loss_D / len(dataloader)
    avg_loss_G = epoch_loss_G / len(dataloader)
    logger.log({"epoch": epoch+1, "loss_D": avg_loss_D, "loss_G": avg_loss_G})
    print(f"Epoch {epoch+1} Completed. Avg_D Loss: {avg_loss_D:.4f}, Avg_G Loss: {avg_loss_G:.4f}", flush=True)

    #save model if improved
    #always saving only the best model (overwrites old one)
    if avg_loss_G < best_loss_G:
        best_loss_G = avg_loss_G
        torch.save(generator.state_dict(), f"checkpoints/{experiment_name}_best_g.pth")
        print(f"Generator improved and saved at epoch {epoch+1}", flush=True)

    if avg_loss_D < best_loss_D:
        best_loss_D = avg_loss_D
        torch.save(discriminator.state_dict(), f"checkpoints/{experiment_name}_best_d.pth")
        print(f"Discriminator improved and saved at epoch {epoch+1}", flush=True)

    #save generated samples every epoch for inspection ===
    with torch.no_grad():
        fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
        generated_samples = generator(fixed_noise)
    save_image(generated_samples, f"images/{experiment_name}_epoch_{epoch+1}.png", normalize=True)

#final save: best models
print("Training finished. Best Generator and Discriminator models already saved.", flush=True)
logger.finish()

