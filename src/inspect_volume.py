# inspect_volume.py
# loads one CT volume, prints its shape, voxel spacing and HU range, then applies
# the same lung windowing Muralidharan (2025) used and saves a slice montage.
# this proves I can read the imaging side, not just the report text.
#
# point VOLUME_PATH at a downloaded .nii.gz once src/download_data.py has pulled a sample.

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUTS_DIR, DATA_DIR, HU_MIN, HU_MAX

import numpy as np
import matplotlib.pyplot as plt


def find_a_volume():
    # grab the first downloaded volume if one exists
    candidates = sorted((DATA_DIR / "ctrate_sample").rglob("*.nii.gz"))
    return candidates[0] if candidates else None


def load_volume(path):
    # nibabel keeps this simple and easy to explain: load, get the array, get spacing
    import nibabel as nib

    img = nib.load(str(path))
    volume = img.get_fdata()
    spacing = img.header.get_zooms()
    return volume, spacing


def report_volume(volume, spacing):
    print("=" * 60)
    print("VOLUME INSPECTION")
    print("=" * 60)
    print(f"shape (voxels):    {volume.shape}")
    print(f"voxel spacing mm:  {tuple(round(float(s), 2) for s in spacing)}")
    print(f"HU min / max:      {volume.min():.0f} / {volume.max():.0f}")
    print(f"HU mean:           {volume.mean():.0f}")


def apply_lung_window(volume):
    # clip to the lung window and scale to [0, 1], same idea as the baseline preprocessing
    windowed = np.clip(volume, HU_MIN, HU_MAX)
    windowed = (windowed - HU_MIN) / (HU_MAX - HU_MIN)
    return windowed


def save_montage(windowed, n_slices=6):
    # pick evenly spaced axial slices through the volume and lay them out in a row
    depth = windowed.shape[2]
    indices = np.linspace(depth * 0.2, depth * 0.8, n_slices).astype(int)

    fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3))
    for ax, idx in zip(axes, indices):
        ax.imshow(windowed[:, :, idx].T, cmap="gray", origin="lower")
        ax.set_title(f"slice {idx}")
        ax.axis("off")
    plt.tight_layout()

    out_path = OUTPUTS_DIR / "sample_volume_montage.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved montage to: {out_path}")


def main():
    path = find_a_volume()
    if path is None:
        print("no CT volume found in data/ctrate_sample/.")
        print("run src/download_data.py with DOWNLOAD_VOLUME_SAMPLE = True first.")
        return

    print(f"inspecting: {path.name}")
    volume, spacing = load_volume(path)
    report_volume(volume, spacing)
    windowed = apply_lung_window(volume)
    save_montage(windowed)


if __name__ == "__main__":
    main()
