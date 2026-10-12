import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import yaml
from debugpy.launcher import channel
from loguru import logger
from scipy.ndimage.filters import gaussian_filter


def load_cfg():
    # Change this line to add an extra .parent
    cfg_path = Path(__file__).parent.parent / "config.yaml"

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)["preprocessing"]

CFG = load_cfg()

#SPATIAL DOMAIN

def nlm_denoise(img_bgr: np.ndarray) -> np.ndarray:
    c = CFG["denoise"]
    c = {
        "enabled":True,
        "h":10,
        "template_window":7,
        "search_window":21
    }
    return cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h = c["h"],
        hColor = c["h"],
        templateWindowSize=c["template_window"],
        searchWindowSize=c["search_window"]
    )

#HISTOGRAM EQUALISATION
def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    c = CFG["clahe"]
    clahe = cv2.createCLAHE(
        clipLimit=c["clip_limit"],
        tileGridSize=tuple(c["tile_grid"])
    )
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

#UNSHARPING MASK
def unsharp_mask(img_bgr: np.ndarray, strength: float | None = None) -> np.ndarray:
    try:
        k = strength or CFG["sharpen"]["kernel_strength"]
        blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=3)
        return cv2.addWeighted(img_bgr, 1.0 + k, blurred, -k, 0)
    except Exception:
        return img_bgr  # return original if sharpening fails

#FREQUENCY DOMAIN

def butterworth_lowpass_mask(rows:int,cols:int,cutoff:float,order:int) -> np.ndarray:
    u = np.fft.fftfreq(rows) * rows
    v = np.fft.fftfreq(cols) * cols
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U ** 2 + V ** 2)
    D0 = cutoff * min(rows, cols) / 2
    H = 1 / (1 + (D / (D0 + 1e-9)) ** (2 * order))
    return np.fft.ifftshift(H)

def freq_domain_filter(img_bgr: np.ndarray) -> np.ndarray:
    c = CFG["freq_domain"]
    order = c["butterworth_order"]
    cutoff = c["cutoff_ratio"]
    result = np.zeros_like(img_bgr,dtype=np.float32)
    rows,cols = img_bgr.shape[:2]
    H = butterworth_lowpass_mask(rows,cols,cutoff,order)
    for ch in range(3):
        channel = img_bgr[:,:,ch].astype(np.float32)
        F = np.fft.fft2(channel)
        F_filt = F*H
        filtered = np.real(np.fft.ifft2(F_filt))
        result[:,:,ch] = np.clip(filtered,0,255)
    return result.astype(np.uint8)

#PIPELINE

def preprocess_image(
        source:str|Path|np.ndarray,
        *,
        return_stages:bool = False,
)->np.ndarray|tuple[np.ndarray,dict]:
    if isinstance(source,(str,Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {source}")
    else:
        img = source.copy()
    stages: dict[str,np.ndarray] = {"original":img.copy()}
    logger.debug(f"Preprocessing image shape: {source.shape}")

    #Denoise->
    if CFG["denoise"]["enabled"]:
        img = nlm_denoise(img)
        stages["denoised"] = img.copy()

    #clahe->
    if CFG["clahe"]["enabled"]:
        img = apply_clahe(img)
        stages["clahe"] = img.copy()

    #Frequency Filter
    if CFG["freq_domain"]["enabled"]:
        img = freq_domain_filter(img)
        stages["freq_filtered"] = img.copy()

    #unsharp mask
    if CFG["sharpen"]["enabled"]:
        img = unsharp_mask(img)
        stages["sharpened"] = img.copy()

    #resize
    target = tuple(CFG["target_size"])
    img = cv2.resize(img,target,interpolation=cv2.INTER_LANCZOS4)
    stages["final"] = img.copy()

    logger.debug("Preprocessing complete")
    if return_stages:
        return img,stages
    return img

def preprocess_batch(
    paths: list[str | Path],
    *,
    progress_callback=None,
)-> list[np.ndarray]:
    results = []
    total   = len(paths)
    for i, p in enumerate(paths):
        try:
            results.append(preprocess_image(p))
        except Exception as exc:
            logger.warning(f"Skipping {p}: {exc}")
            results.append(None)
        if progress_callback:
            progress_callback(i + 1, total)
    return results

def bgr_to_pil(img_bgr:np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def pil_to_bgr(img_pil:Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)




