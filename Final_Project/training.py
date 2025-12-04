import os
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import ResNet50
from dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import get_device, create_log_dir, plot_metrics, compute_mean_std

# options
torch.manual_seed(42)

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        running_correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    avg_acc = running_correct / total
    return avg_loss, avg_acc


# def evaluate(model, dataloader, loss_fn, device):
#     model.eval()
#     running_loss = 0.0
#     running_correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
#             images, labels = images.to(device), labels.to(device)

#             outputs = model(images)
#             loss = loss_fn(outputs, labels)

#             running_loss += loss.item() * labels.size(0)
#             _, preds = outputs.max(1)
#             running_correct += preds.eq(labels).sum().item()
#             total += labels.size(0)

#     avg_loss = running_loss / total
#     avg_acc = running_correct / total
#     return avg_loss, avg_acc

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * labels.size(0)

            _, preds = outputs.max(1)
            running_correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds)
            all_labels.append(labels)

    avg_loss = running_loss / total
    avg_acc = running_correct / total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # micro-average 계산
    TP = (all_preds == all_labels).sum().item()

    # multi-class는 FP, FN을 다음과 같이 정의
    FP = (all_preds != all_labels).sum().item()
    FN = FP  # micro에서는 FP 총합 == FN 총합

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1_score  = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, avg_acc, precision, recall, f1_score



def training(early_stopping=True, ema_patience=5, learning_rate=1e-5, weight_decay=1e-5, max_epochs=50):
    device = get_device()
    print(f"Using device: {device}")

    # dataset & loader
    root_dir = "/Users/jungyu_park/Desktop/DGU/4_2/Computer Vision/Final_Project"
    dataset_dir = os.path.join(root_dir, "POC_Dataset")
    log_dir = create_log_dir(root_dir)
    
    if early_stopping:
        train_dataset = CustomImageDataset(dataset_dir, mode="Training", transform=None, early_stopping=early_stopping)
        val_dataset = CustomImageDataset(dataset_dir, mode="Training", transform=None, train_dataset=train_dataset)
    else:
        train_dataset = CustomImageDataset(dataset_dir, mode="Training", transform=None)
        val_dataset = None

    mean, std = compute_mean_std(train_dataset)

    # data augmentation transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset.transform = train_transform
    if val_dataset is not None:
        val_dataset.transform = transform

    test_dataset = CustomImageDataset(dataset_dir, mode="Testing", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Logs saved to: {log_dir}")


    # model load & setup
    num_classes = len(train_dataset.classes)
    model = ResNet50(num_classes=num_classes).to(device)
    max_epochs = max_epochs

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # log setup
    log_file = os.path.join(log_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write("Training Log\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Max Epochs: {max_epochs}\n")
        f.write(f"Early Stopping: {early_stopping}\n")
        f.write(f"EMA Patience: {ema_patience}\n")
        f.write("epoch\ttrain_loss\ttrain_acc\tvalid_loss\tvalid_acc\n")

    train_loss_list = []
    train_acc_list = []

    valid_loss_list = []
    valid_acc_list = []

    # training loop
    print("Training started...")

    # early stopping algorithm with Exponential Moving Average (EMA)
    best_model_state = None
    best_loss = None
    patience = ema_patience
    patience_counter = 0

    cur_epoch = 0
    for epoch in range(1, max_epochs + 1):
        cur_epoch += 1
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        if val_loader is not None:
            valid_loss, valid_acc, _, _, _ = evaluate(model, val_loader, loss_fn, device)

            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)

            if best_loss is None:
                best_loss = valid_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        scheduler.step()

        # save epoch log
        with open(log_file, "a") as f:
            if val_loader is None:
                f.write(f"{epoch}\t{train_loss:.4f}\t{train_acc:.4f}\tN/A\tN/A\n")
                valid_loss, valid_acc = 0.0, 0.0
            else:
                f.write(f"{epoch}\t{train_loss:.4f}\t{train_acc:.4f}\t{valid_loss:.4f}\t{valid_acc:.4f}\n")

        if val_loader is None:
            print(f"[Epoch {epoch}/{max_epochs}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        else:
            print(f"[Epoch {epoch}/{max_epochs}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  Valid Loss: {valid_loss:.4f}  Valid Acc: {valid_acc:.4f}")

    if cur_epoch < max_epochs:
        # early stopped
        print(f"Training stopped early at epoch {cur_epoch}.")
    else:
        print("Training completed.")

    # save model
    model_path = os.path.join(log_dir, "resnet50_trained.pth")
    if best_model_state is None:
        best_model_state = model.state_dict()
    torch.save(best_model_state, model_path)
    print(f"Model saved to: {model_path}")

    # plot metrics
    plot_metrics(log_dir, train_loss_list, train_acc_list, range(1, cur_epoch+1), mode="Train")
    if val_loader is not None:
        plot_metrics(log_dir, valid_loss_list, valid_acc_list, range(1, cur_epoch+1), mode="Valid")

    # final evaluation
    model.load_state_dict(best_model_state)
    test_loss, test_acc, _, _, _ = evaluate(model, test_loader, loss_fn, device)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Acc: {test_acc:.4f}")

    with open(log_file, "a") as f:
        f.write(f"Final Test Loss: {test_loss:.4f}, Final Test Acc: {test_acc:.4f}\n")


if __name__ == "__main__":

    # early_sopping_list = [False, True, True, True, True, True, False, True]
    # EMA_patience_list = [0, 5, 5, 10, 5, 5, 0, 10]
    # learning_rate_list = [1e-4, 1e-4, 3e-4, 5e-5, 3e-4, 1e-4, 1e-4, 3e-4]
    # weight_decay_list = [1e-4, 5e-5, 1e-4, 1e-5, 5e-5, 1e-5, 5e-5, 5e-5]
    # max_epochs_list = [50, 90, 90, 90, 90, 90, 50, 90]

    early_sopping_list = [True]
    EMA_patience_list = [5]
    learning_rate_list = [3e-4]
    weight_decay_list = [1e-4]
    max_epochs_list = [90]

    for early_stopping, ema_patience, learning_rate, weight_decay, max_epochs in zip(early_sopping_list, EMA_patience_list, learning_rate_list, weight_decay_list, max_epochs_list):
        print(f"Starting training with early_stopping={early_stopping}, ema_patience={ema_patience}, learning_rate={learning_rate}, weight_decay={weight_decay}, max_epochs={max_epochs}")
        training(early_stopping=early_stopping, ema_patience=ema_patience, learning_rate=learning_rate, weight_decay=weight_decay, max_epochs=max_epochs)