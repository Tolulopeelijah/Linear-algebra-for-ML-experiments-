import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.datasets import load_sample_image


# Try loading a local image first
image_path = "images/tolu.jpg"

if os.path.exists(image_path):
    # Load image from disk
    img = Image.open(image_path)
    print(f"Loaded image: {img.size[0]}×{img.size[1]}, mode={img.mode}")

    # Working in grayscale keeps things easier to reason about
    if img.mode != "L":
        img_gray = img.convert("L")
    else:
        img_gray = img

    image_array = np.array(img_gray, dtype=np.float32) / 255.0
    original_size_bytes = os.path.getsize(image_path)

else:
    # Fall back to a standard sample image if the file isn't found
    print(f"Image not found at '{image_path}'. Using a sample flower image instead.")

    flower = load_sample_image("flower.jpg")
    img_gray = Image.fromarray(flower).convert("L")

    image_array = np.array(img_gray, dtype=np.float32) / 255.0
    original_size_bytes = image_array.size  # rough proxy for raw size


print(f"Image array shape: {image_array.shape}")
print(f"Matrix rank: {np.linalg.matrix_rank(image_array)}")
print(f"Approx. original size: {original_size_bytes/1024:.1f} KB")


# Singular Value Decomposition
U, S, Vt = np.linalg.svd(image_array, full_matrices=False)

BYTES_PER_VALUE = 4  # assuming float32 storage
m, n = image_array.shape
original_pixels = m * n
original_raw_bytes = original_pixels * BYTES_PER_VALUE


def svd_storage(k, m, n, bytes_per_value=4):
    """
    Storage required for a rank-k approximation.

    We keep:
    - U: m × k
    - singular values: k
    - V: n × k
    """
    total_values = k * (m + n + 1)
    return total_values, total_values * bytes_per_value


print("\n" + "=" * 70)
print("LOW-RANK APPROXIMATIONS")
print("=" * 70)

ranks_to_test = [1, 5, 10, 20, 30, 50, 100, 150, 200]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

# Original image reference
axes[0].imshow(image_array, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")


for idx, k in enumerate(ranks_to_test[1:], start=1):
    if idx >= len(axes):
        break

    # Rebuild the image using only the first k singular components
    reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    total_values, total_bytes = svd_storage(k, m, n, BYTES_PER_VALUE)
    compression_ratio = original_raw_bytes / total_bytes

    mse = np.mean((image_array - reconstructed) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    energy = np.sum(S[:k] ** 2) / np.sum(S ** 2)

    axes[idx].imshow(np.clip(reconstructed, 0, 1), cmap="gray")
    axes[idx].set_title(
        f"Rank {k}\n"
        f"{total_bytes/1024:.1f} KB\n"
        f"{compression_ratio:.1f}× smaller\n"
        f"PSNR {psnr:.1f} dB"
    )
    axes[idx].axis("off")

    print(f"Rank {k:3d} | "
          f"Storage: {total_bytes/1024:6.1f} KB | "
          f"Compression: {compression_ratio:5.1f}× | "
          f"Energy kept: {energy:6.2%} | "
          f"PSNR: {psnr:5.1f} dB")


plt.tight_layout()
plt.show()


# Rate–distortion view: how quality improves as we spend more storage
storage_kb = []
psnr_values = []

ranks = list(range(1, min(m, n), max(1, min(m, n) // 20)))

for k in ranks[:50]:
    _, total_bytes = svd_storage(k, m, n, BYTES_PER_VALUE)
    reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    mse = np.mean((image_array - reconstructed) ** 2)

    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

    storage_kb.append(total_bytes / 1024)
    psnr_values.append(min(psnr, 100))


plt.figure(figsize=(10, 5))
plt.plot(storage_kb, psnr_values, marker="o")
plt.xlabel("Storage (KB)")
plt.ylabel("PSNR (dB)")
plt.title("Rate–Distortion Curve: SVD Image Approximation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
