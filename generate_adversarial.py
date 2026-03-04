import os
import cv2
import random
import numpy as np

INPUT_DIR = "data/test"

def add_noise(frame, level):
    if level == 1:
        std = 10
    elif level == 2:
        std = 25
    else:
        std = 45

    noise = np.random.normal(0, std, frame.shape).astype(np.uint8)
    return cv2.add(frame, noise)


def add_blur(frame, level):
    if level == 1:
        k = 5
    elif level == 2:
        k = 11
    else:
        k = 21

    return cv2.GaussianBlur(frame, (k, k), 0)


def process_video(input_path, output_path, level):
    cap = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        choice = random.choice(["noise", "blur"])

        if choice == "noise":
            frame = add_noise(frame, level)
        else:
            frame = add_blur(frame, level)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    print(f"Processed {os.path.basename(input_path)} | Level {level} | Frames: {frame_count}")


# Generate 3 levels of perturbation
for level in [1, 2, 3]:
    print(f"\nGenerating Level {level} adversarial data...")

    for cls in ["real", "fake"]:
        input_cls = os.path.join(INPUT_DIR, cls)
        output_cls = os.path.join(f"data/test_adv_l{level}", cls)

        os.makedirs(output_cls, exist_ok=True)

        for file in os.listdir(input_cls):
            if file.endswith(".mp4"):
                input_path = os.path.join(input_cls, file)
                output_path = os.path.join(output_cls, file)
                process_video(input_path, output_path, level)

print("\nAll adversarial levels created successfully.")