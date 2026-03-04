import os
import shutil
import random

random.seed(42)

base_dir = "data"
classes = ["real", "fake"]

split_ratio = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

for cls in classes:
    cls_path = os.path.join(base_dir, cls)
    files = os.listdir(cls_path)
    random.shuffle(files)

    total = len(files)
    train_end = int(total * split_ratio["train"])
    val_end = train_end + int(total * split_ratio["val"])

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split_name, split_files in splits.items():
        split_folder = os.path.join(base_dir, split_name, cls)
        os.makedirs(split_folder, exist_ok=True)

        for file in split_files:
            src = os.path.join(cls_path, file)
            dst = os.path.join(split_folder, file)
            shutil.move(src, dst)

print("Dataset split complete.")