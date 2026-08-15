import os
import imageio.v2 as imageio
# import imageio
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Convert images to video')
parser.add_argument('--image_folder', type=str, help='Path of image folder')
parser.add_argument('--video_path', type=str, help='Video filename')
parser.add_argument('--fps', type=float, default=15.0, help='Frame per second')
args = parser.parse_args()

image_folder = args.image_folder
video_path = args.video_path
fps = float(args.fps)

video_folder = os.path.dirname(video_path)
os.makedirs(video_folder, exist_ok=True)

images_path = sorted(
    [
        img
        for img in os.listdir(image_folder)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)
if len(images_path) == 0:
    print("No images found in the folder")

frame_series = []
for image_path in images_path:
    image = imageio.imread(os.path.join(image_folder, image_path)).astype(np.uint8)
    h = image.shape[0] if image.shape[0] % 2 == 0 else image.shape[0] - 1
    w = image.shape[1] if image.shape[1] % 2 == 0 else image.shape[1] - 1
    frame_series.append(image[:h, :w])

imageio.mimsave(video_path, frame_series, fps=fps, macro_block_size=1)
