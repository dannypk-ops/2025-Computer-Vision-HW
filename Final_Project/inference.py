import os
from tqdm import tqdm

from model import ResNet50
from utils import get_device
from training import evaluate
from dataset import CustomImageDataset

import torch
import torch.nn as nn
from torchvision import transforms

def load_model(model_path, device):
    model = ResNet50(num_classes=4) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__": 
    model_path = "Final_Project/logs_2025-12-03_23-11-12/resnet50_trained.pth"
    data_root = "Final_Project/POC_Dataset"
    device = get_device()
    model = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5344, 0.6485, 0.6477], std=[0.1589, 0.1876, 0.1327])
    ])

    test_dataset = CustomImageDataset(data_root, mode="Testing", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_precision, test_recall, test_f1_score = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}%")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1_score:.4f}")  