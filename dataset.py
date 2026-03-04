import os
import cv2
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=10, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.samples = []
        self.transform = transform

        classes = ["real", "fake"]

        for label, cls in enumerate(classes):
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith(".mp4"):
                    self.samples.append(
                        (os.path.join(cls_path, file), label)
                    )

    def __len__(self):
        return len(self.samples)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = sorted(
            random.sample(range(total_frames), 
                          min(self.num_frames, total_frames))
        )

        frames = []
        current_frame = 0
        selected_idx = 0

        while cap.isOpened() and selected_idx < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame == frame_indices[selected_idx]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                selected_idx += 1

            current_frame += 1

        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.extract_frames(video_path)

        processed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            processed_frames.append(frame)

        # Stack frames → (num_frames, 3, H, W)
        frames_tensor = torch.stack(processed_frames)

        return frames_tensor, torch.tensor(label, dtype=torch.long)