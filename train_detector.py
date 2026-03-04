import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from dataset import DeepfakeVideoDataset
from utils.transforms import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 2
NUM_FRAMES = 5
EPOCHS = 5
LR = 1e-4

# Datasets
train_dataset = DeepfakeVideoDataset(
    root_dir="data/train",
    num_frames=NUM_FRAMES,
    transform=get_transforms()
)

val_dataset = DeepfakeVideoDataset(
    root_dir="data/val",
    num_frames=NUM_FRAMES,
    transform=get_transforms()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train_one_epoch():
    model.train()
    total_loss = 0

    for videos, labels in train_loader:
        videos = videos.to(device)
        labels = labels.to(device)

        batch_size, num_frames, C, H, W = videos.shape
        videos = videos.view(-1, C, H, W)

        outputs = model(videos)
        outputs = outputs.view(batch_size, num_frames, -1)
        outputs = outputs.mean(dim=1)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            batch_size, num_frames, C, H, W = videos.shape
            videos = videos.view(-1, C, H, W)

            outputs = model(videos)
            outputs = outputs.view(batch_size, num_frames, -1)
            outputs = outputs.mean(dim=1)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

for epoch in range(EPOCHS):
    loss = train_one_epoch()
    acc = evaluate()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Val Accuracy: {acc:.2f}%")

torch.save(model.state_dict(), "models/resnet_baseline.pth")
print("Model saved.")