# plots.py: figures.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def topdown(trajs, path, title=""):
    """trajs: dict name -> N x 4 x 4 camera-to-world poses. Plots x (right) against z (forward), seen from above."""
    plt.figure(figsize=(6, 6))
    for name, T in trajs.items():
        p = np.asarray(T)[:, :3, 3]
        plt.plot(p[:, 0], p[:, 2], label=name, lw=2 if name == "ground truth" else 1.2)
    plt.axis("equal"); plt.xlabel("x (m)"); plt.ylabel("z (m)"); plt.legend(); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def scale_over_time(series, path):
    """series: dict name -> per-step scale (m)."""
    plt.figure(figsize=(8, 3.5))
    for name, s in series.items():
        plt.plot(s, label=name, lw=1.2)
    plt.xlabel("frame"); plt.ylabel("step length (m)"); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def hist(values, path, xlabel, bins=40):
    plt.figure(figsize=(5, 3.5)); plt.hist(values, bins=bins); plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
