import os
import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class FFHQDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        # Use recursive glob to capture all PNG images in the directory and subdirectories
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '**/*.png'), recursive=True))
        # Define the transformation: resize, convert to Tensor, and normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

