import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FFHQDataset
from model import VAE
from utils import vae_loss, load_config
from logger import Logger

#device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#config file from CLI
if len(sys.argv) != 2:
    print("Usage: python train.py <config_path>")
    sys.exit(1)

config_path = sys.argv[1]
config = load_config(config_path)

#set dataset path
dataset_root = (
    config["data_settings"]["hyperion_dir"]
    if config["data_settings"].get("use_hyperion", False)
    else config["data_settings"]["root_dir"]
)

#prepare Logger
experiment_name = config["experiment_settings"]["experiment_name"]
logger = Logger(experiment_name=experiment_name)

#load dataset
dataset = FFHQDataset(dataset_root, config["data_settings"]["image_size"])
dataloader = DataLoader(
    dataset,
    batch_size=config["train_settings"]["batch_size"],
    shuffle=True,
    num_workers=config["train_settings"].get("num_workers", 4),
)

#initialize model
vae_model = VAE(
    in_channels=config["model_settings"]["in_channels"],
    latent_dim=config["train_settings"]["latent_dim"],
).to(device)

#use DataParallel for multi-GPU setup
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!", flush=True)
    vae_model = torch.nn.DataParallel(vae_model)

optimizer = optim.Adam(
    vae_model.parameters(),
    lr=config["train_settings"]["learning_rate"],
    weight_decay=1e-4,
)

#early stopping settings
patience = config["train_settings"].get("early_stopping_patience", 20)
best_loss = float("inf")
patience_counter = 0

num_epochs = config["train_settings"]["num_epochs"]

#training loop
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}/{num_epochs}", flush=True)
    epoch_loss = 0.0
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_x, mu, log_var = vae_model(batch)
        loss = vae_loss(recon_x, batch, mu, log_var, config["train_settings"]["beta"])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        if i % 50 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(dataloader)}] Loss: {loss.item():.4f}", flush=True)
    
    avg_loss = epoch_loss / len(dataloader)
    logger.log({"epoch": epoch + 1, "loss": avg_loss})
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.2f}", flush=True)

    #check for early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(vae_model.state_dict(), f"{experiment_name}_best.pth")
        print(f"Model improved, saving best model for epoch {epoch+1}", flush=True)
    else:
        patience_counter += 1
        print(f"No improvement at epoch {epoch+1}. Patience counter: {patience_counter}/{patience}", flush=True)
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}", flush=True)
            break

#final save
torch.save(vae_model.state_dict(), f"{experiment_name}.pth")
logger.save_model(vae_model, f"{experiment_name}.pth")
print("Training complete. Final model saved.", flush=True)
logger.finish()