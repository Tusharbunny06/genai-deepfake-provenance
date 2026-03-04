import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import DeepfakeVideoDataset
from utils.transforms import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_dataset = DeepfakeVideoDataset(
    root_dir="data/test_adv_l3",
    num_frames=5,
    transform=get_transforms()
)

test_loader = DataLoader(test_dataset, batch_size=1)

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/resnet_baseline.pth"))
model = model.to(device)
model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for videos, labels in test_loader:
        videos = videos.to(device)
        labels = labels.to(device)

        batch_size, num_frames, C, H, W = videos.shape
        videos = videos.view(-1, C, H, W)

        outputs = model(videos)
        outputs = outputs.view(batch_size, num_frames, -1)
        outputs = outputs.mean(dim=1)

        probs = torch.softmax(outputs, dim=1)[:,1]

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Metrics
print(classification_report(all_labels, all_preds))

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

roc_auc = roc_auc_score(all_labels, all_probs)
print("ROC-AUC:", roc_auc)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()