import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, root, mode = 'Training', transform=None, train_dataset=None, early_stopping=False):
        self.root = root

        self.ratio = 0.9
        self.early_stopping = early_stopping
        self.mode = mode

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transform

        if self.mode == 'Training':
            self.data_path = os.path.join(self.root, 'Training')
        else:
            self.data_path = os.path.join(self.root, 'Testing')

        self.samples = []
        
        all_entries = os.listdir(self.data_path)
        self.classes = sorted([c for c in all_entries if os.path.isdir(os.path.join(self.data_path, c))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.train_samples_paths = None
        if train_dataset is not None:
            self.train_samples_paths = set([sample[0] for sample in train_dataset.samples]) 

        all_potential_samples = []
        for cls_name in self.classes:
            cls_path = os.path.join(self.data_path, cls_name)
            if not os.path.isdir(cls_path): continue

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                
                if self.train_samples_paths is not None:
                    if img_path in self.train_samples_paths:
                        continue
                
                all_potential_samples.append((img_path, self.class_to_idx[cls_name]))

        if self.mode == 'Training' and self.early_stopping:
            random.seed(42)
            random.shuffle(all_potential_samples)

            train_size = int(self.ratio * len(all_potential_samples))
            self.samples = all_potential_samples[:train_size]
            
        else:
            self.samples = all_potential_samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label