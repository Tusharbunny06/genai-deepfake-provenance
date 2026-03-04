import imageio
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

# load model once
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")


def generate_video(prompt):

    # generate AI image
    image = pipe(
        prompt,
        num_inference_steps=20
    ).images[0]

    width, height = image.size

    frames = []

    img_np = np.array(image)

    for i in range(12):

        frame = np.roll(img_np, i * 5, axis=1)  # simple animation
        frames.append(frame)

    video_path = "generated_video.mp4"

    writer = imageio.get_writer(video_path, fps=3)

    for frame in frames:
        writer.append_data(frame)

    writer.close()

    return video_path