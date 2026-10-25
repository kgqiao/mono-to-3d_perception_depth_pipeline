# corrupt.py (D7): fog, blur and brightness at 3 severities, applied to a sequence folder.
#   python corrupt.py --images data/kitti/sequences/00/image_0 --out data/corrupt/00 --n 300
# Writes <out>/<kind>_<severity>/<frame>.png. Uses the imagecorruptions library when installed, else simple OpenCV versions.
# These are SYNTHETIC corruptions. Do not claim real-weather results.
import argparse
import glob
import os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
try:
    from imagecorruptions import corrupt as ic
except ImportError:
    ic = None


def simple(img, kind, sev):
    f = img.astype(np.float32)
    if kind == "fog":                                   # blend toward grey, more at the top (far away)
        h = img.shape[0]; w = np.linspace(0.6, 0.15, h)[:, None] * (0.6 + 0.4 * sev)
        if img.ndim == 3: w = w[..., None]
        f = f * (1 - w) + 200 * w
    elif kind == "blur":
        k = 2 * (2 * sev) + 1; f = cv2.GaussianBlur(f, (k, k), 0)
    elif kind == "brightness":
        f = f * (1 - 0.25 * sev)                        # darker, like dusk
    return np.clip(f, 0, 255).astype(np.uint8)


NAMES = {"fog": "fog", "blur": "gaussian_blur", "brightness": "brightness"}
files = sorted(glob.glob(os.path.join(a.images, "*.png")))
files = files[:a.n] if a.n else files
for kind in NAMES:
    for sev in (1, 2, 3):
        d = os.path.join(a.out, f"{kind}_{sev}"); os.makedirs(d, exist_ok=True)
        for f in files:
            img = cv2.imread(f)
            out = ic(img, corruption_name=NAMES[kind], severity=sev) if ic else simple(img, kind, sev)
            cv2.imwrite(os.path.join(d, os.path.basename(f)), out)
print(f"wrote {len(files)} frames x 9 settings to {a.out} ({'imagecorruptions' if ic else 'OpenCV fallback'})")
