# depth_infer.py: run a metric Depth Anything V2 checkpoint and save raw float depth.
#   python depth_infer.py --images data/kitti/sequences/00/image_2 --out out/depth/00 --n 50
# Saves <frame>.npy in meters (float32) plus a 16-bit preview PNG. Save the .npy, not just the preview.
import argparse
import glob
import os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
ap.add_argument("--n", type=int, default=0, help="0 = all")
a = ap.parse_args()

from transformers import pipeline          # imported late so --help works without torch installed

pipe = pipeline("depth-estimation", model=a.model)
os.makedirs(a.out, exist_ok=True)
files = sorted(glob.glob(os.path.join(a.images, "*.png")))
files = files[:a.n] if a.n else files

for f in files:
    img = Image.open(f).convert("RGB")
    res = pipe(img)
    d = res["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)

    if d.shape != (img.height, img.width):  # the model works at its own size; resize back to the image
        d = np.asarray(Image.fromarray(d).resize((img.width, img.height), Image.BILINEAR), dtype=np.float32)
    name = os.path.splitext(os.path.basename(f))[0]

    np.save(os.path.join(a.out, name + ".npy"), d)
    
    Image.fromarray((np.clip(d / 80.0, 0, 1) * 65535).astype(np.uint16)).save(os.path.join(a.out, name + "_preview.png"))

print(f"saved {len(files)} depth maps to {a.out} (meters, float32 .npy)")
