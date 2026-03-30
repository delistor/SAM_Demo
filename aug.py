# industrial_augmentation.py
import cv2
import numpy as np

# -----------------------------
# 1️⃣ Gamma Correction
# -----------------------------
def gamma_correction(img, gamma=None):
    """
    img: uint8 BGR image
    gamma: float or None, if None random in [0.7, 1.5]
    """
    if gamma is None:
        gamma = np.random.uniform(0.7, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


# -----------------------------
# 2️⃣ Local Illumination
# -----------------------------
def local_illumination(img, strength=0.5):
    h, w = img.shape[:2]
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)

    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    sigma = np.random.uniform(0.3, 0.7) * min(h, w)
    mask = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    mask = (mask - mask.min()) / (mask.max() - mask.min())

    factor = 1.0 + strength * (np.random.rand() * 2 - 1)
    mask = 1 + (mask - 0.5) * factor
    mask = np.expand_dims(mask, axis=2)

    out = img.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------------
# 3️⃣ Noise
# -----------------------------
def add_gaussian_noise(img, sigma=10):
    noise = np.random.randn(*img.shape) * sigma
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_speckle_noise(img, sigma=0.1):
    noise = np.random.randn(*img.shape) * sigma
    out = img.astype(np.float32) + img.astype(np.float32) * noise
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------------
# 4️⃣ Blur
# -----------------------------
def gaussian_blur(img):
    k = np.random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img, max_kernel=15):
    k = np.random.randint(5, max_kernel)
    kernel = np.zeros((k, k))
    if np.random.rand() < 0.5:
        kernel[k//2, :] = np.ones(k)
    else:
        kernel[:, k//2] = np.ones(k)
    kernel = kernel / k
    return cv2.filter2D(img, -1, kernel)


# -----------------------------
# 5️⃣ Contrast / Brightness / CLAHE
# -----------------------------
def brightness_contrast(img):
    alpha = np.random.uniform(0.7, 1.3)  # contrast
    beta = np.random.uniform(-30, 30)    # brightness
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_enhancement(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# -----------------------------
# 6️⃣ Combined augmentation pipeline
# -----------------------------
def augment(img):
    """
    img: uint8 BGR image
    returns: augmented image
    """
    out = img.copy()

    if np.random.rand() < 0.7:
        out = gamma_correction(out)

    if np.random.rand() < 0.7:
        out = local_illumination(out)

    if np.random.rand() < 0.5:
        out = add_gaussian_noise(out)

    if np.random.rand() < 0.5:
        out = gaussian_blur(out)

    if np.random.rand() < 0.5:
        out = brightness_contrast(out)

    if np.random.rand() < 0.3:
        out = clahe_enhancement(out)

    return out


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    img = cv2.imread("wafer.png")
    aug = augment(img)
    cv2.imwrite("wafer_aug.png", aug)