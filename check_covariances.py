import numpy as np
import glob
import os
import matplotlib.pyplot as plt

files = sorted(glob.glob("covariances/*.npy"))

print(f"Found {len(files)} covariance files\n")

for fname in files:
    cov = np.load(fname)

    print(os.path.basename(fname))
    print("  shape:", cov.shape)
    print("  all finite:", np.all(np.isfinite(cov)))
    print("  min:", np.min(np.abs(cov)))
    print("  max:", np.max(np.abs(cov)))
    print("  mean:", np.mean(np.abs(cov)))
    print()

    plt.figure(figsize=(5, 4))
    plt.imshow(
        np.fft.fftshift(np.abs(cov)),
        origin="lower"
    )
    plt.colorbar(label="|covariance|")
    plt.title(os.path.basename(fname))
    plt.tight_layout()
    plt.show()