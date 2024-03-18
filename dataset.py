import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageTextTranslationDataset(Dataset):
    def __init__(self, img_paths, texts, transform=None):
        self.img_paths = img_paths
        self.texts = texts
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        text = self.texts[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)

        return img, text
