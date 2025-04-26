import sys
import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from utils import load_config

config_file = sys.argv[1]
config = load_config(config_file)

#switch path depending on environment
if config["data_settings"].get("use_hyperion", False):
    dataset_root = config["data_settings"]["hyperion_dir"]
else:
    dataset_root = config["data_settings"]["root_dir"]

class FFHQDataset(Dataset):
    def __init__(self, root_dir, img_size=64):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img