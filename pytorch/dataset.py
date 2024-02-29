import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class DogsvsCatsDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        super().__init__()
        self.transform = transform or transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.image_folder = image_folder
        self.image_files = os.listdir(image_folder)
        self.labels = torch.tensor(
            [0 if "dog" in filename else 1 for filename in self.image_files])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[index]
        return image, label
