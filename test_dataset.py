from dataset import DeepfakeVideoDataset
from utils.transforms import get_transforms

dataset = DeepfakeVideoDataset(
    root_dir="data/train",
    num_frames=5,
    transform=get_transforms()
)

print("Total samples:", len(dataset))

sample, label = dataset[0]

print("Sample shape:", sample.shape)
print("Label:", label)