import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from components.ASID import ASID
from components.ASID_PreUpsample import ASID_PreUpsample
from utilities.utilities import tensor2img, calculate_psnr, calculate_ssim
from utilities.json_config import readConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR_PATH = "benchmark/LR/LRBI/Urban100/x8/img_001_LRBI_x8.png"
HR_PATH = "benchmark/HR/Urban100/x8/img_001_HR_x8.png"

OUT_DIR = "comparison_img001"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(config_path, checkpoint_path, model_class):

    cfg = readConfig(config_path)

    model = model_class(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=cfg["feature_num"],
        **cfg["module_params"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove profiling-only parameters stored in the checkpoint
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.endswith("total_ops")
        and not k.endswith("total_params")
    }

    model.load_state_dict(state_dict, strict=True)

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(path):

    img = np.array(
        Image.open(path).convert("RGB")
    ).astype(np.float32)

    tensor = torch.from_numpy(
        img.transpose(2, 0, 1)
    ).unsqueeze(0)

    # Repository normalization
    tensor = (tensor / 255.0 - 0.5) * 2.0

    return tensor.to(DEVICE)


# ============================================================
# SAVE TENSOR AS IMAGE
# ============================================================

def save_tensor(tensor, filename):

    image = tensor2img(tensor.cpu())[0]

    # PIL needs uint8 for RGB PNG output
    image = np.clip(image, 0, 255).astype(np.uint8)

    Image.fromarray(image).save(
        os.path.join(OUT_DIR, filename)
    )

    return image


# ============================================================
# PATCH INFERENCE
# Same strategy as repository evaluator
# ============================================================

def patch_inference(model, lr, scale=8):

    tile = 64
    overlap = 24

    b, c, h, w = lr.shape

    tile = min(tile, h, w)
    stride = tile - overlap

    h_indices = list(
        range(0, h - tile, stride)
    ) + [h - tile]

    w_indices = list(
        range(0, w - tile, stride)
    ) + [w - tile]

    output = torch.zeros(
        b, c, h * scale, w * scale,
        device=lr.device
    )

    weight = torch.zeros_like(output)

    with torch.no_grad():

        for hi in h_indices:

            for wi in w_indices:

                patch = lr[
                    ...,
                    hi:hi + tile,
                    wi:wi + tile
                ]

                result = model(patch)

                if isinstance(result, list):
                    result = result[-1]

                output[
                    ...,
                    hi * scale:(hi + tile) * scale,
                    wi * scale:(wi + tile) * scale
                ] += result

                weight[
                    ...,
                    hi * scale:(hi + tile) * scale,
                    wi * scale:(wi + tile) * scale
                ] += 1

    return output / weight


# ============================================================
# MAIN
# ============================================================

print("=" * 75)
print("       ASID ×8 STEP-BY-STEP VISUAL COMPARISON")
print("=" * 75)

print("Device:", DEVICE)
print()
print("Image: Urban100 img_001")
print()
print("Pipeline:")
print("HR → LR ×8 → Bilinear ×2 → ASID → PixelShuffle ×4")
print("=" * 75)


# ============================================================
# STEP 1 — ORIGINAL HR
# ============================================================

print("\n[STEP 1/6] Loading Ground Truth HR...")

hr = load_image(HR_PATH)

hr_img = save_tensor(
    hr,
    "01_ground_truth_HR.png"
)

print("✓ Ground Truth HR saved")
print("  Size:", hr_img.shape[1], "x", hr_img.shape[0])


# ============================================================
# STEP 2 — INPUT LR
# ============================================================

print("\n[STEP 2/6] Loading LR ×8 input...")

lr = load_image(LR_PATH)

lr_img = save_tensor(
    lr,
    "02_input_LR_x8.png"
)

print("✓ LR input saved")
print("  Size:", lr_img.shape[1], "x", lr_img.shape[0])


# ============================================================
# STEP 3 — BILINEAR ×2
# ============================================================

print("\n[STEP 3/6] Applying Bilinear ×2...")

with torch.no_grad():

    bilinear_2x = F.interpolate(
        lr,
        scale_factor=2,
        mode="bilinear",
        align_corners=False
    )

bilinear_img = save_tensor(
    bilinear_2x,
    "03_bilinear_x2.png"
)

print("✓ Bilinear ×2 image saved")
print(
    "  Size:",
    bilinear_img.shape[1],
    "x",
    bilinear_img.shape[0]
)


# ============================================================
# IMPORTANT:
# The trained PreUpsample model itself performs the internal
# Bilinear ×2 operation.
#
# Therefore, we DO NOT feed bilinear_2x into the model.
# We feed the ORIGINAL LR image.
# ============================================================

# ============================================================
# STEP 4 — PROPOSED MODEL
# ============================================================

print("\n[STEP 4/6] Running Proposed PreUpsample ASID...")

print("  Internal pipeline:")
print("  LR → internal Bilinear ×2 → ASID → PixelShuffle ×4")

model_A = load_model(
    "train_logs/ASID_PreUpsample_X8_DIV2K/model_config.json",
    "../checkpoints/ASID_PreUpsample_X8_DIV2K/epoch20_ASID.pth",
    ASID_PreUpsample
)

output_A = patch_inference(
    model_A,
    lr,
    scale=8
)

img_A = save_tensor(
    output_A,
    "04_proposed_ASID_x4_final_x8.png"
)

print("✓ Proposed ASID output saved")
print(
    "  Final size:",
    img_A.shape[1],
    "x",
    img_A.shape[0]
)


# ============================================================
# STEP 5 — DIRECT ASID ×8
# ============================================================

print("\n[STEP 5/6] Running Direct ASID ×8...")

print("  Pipeline:")
print("  LR → ASID → PixelShuffle ×8")

model_B = load_model(
    "train_logs/ASID_X8_DIV2K/model_config.json",
    "../checkpoints/ASID_X8_DIV2K/epoch20_ASID.pth",
    ASID
)

output_B = patch_inference(
    model_B,
    lr,
    scale=8
)

img_B = save_tensor(
    output_B,
    "05_direct_ASID_x8.png"
)

print("✓ Direct ASID output saved")
print(
    "  Final size:",
    img_B.shape[1],
    "x",
    img_B.shape[0]
)


# ============================================================
# STEP 6 — METRICS
# ============================================================

print("\n[STEP 6/6] Calculating PSNR and SSIM...")

hr_image = hr_img

psnr_A = calculate_psnr(
    img_A,
    hr_image
)

ssim_A = calculate_ssim(
    img_A,
    hr_image
)

psnr_B = calculate_psnr(
    img_B,
    hr_image
)

ssim_B = calculate_ssim(
    img_B,
    hr_image
)


print()
print("=" * 75)
print("                    FINAL COMPARISON")
print("=" * 75)

print()
print("PROPOSED — Bilinear ×2 + ASID ×4")
print("-----------------------------------")
print("PSNR :", round(psnr_A, 4), "dB")
print("SSIM :", round(ssim_A, 4))

print()
print("BASELINE — Direct ASID ×8")
print("-----------------------------------")
print("PSNR :", round(psnr_B, 4), "dB")
print("SSIM :", round(ssim_B, 4))

print()
print("IMPROVEMENT (Proposed - Baseline)")
print("-----------------------------------")
print("PSNR :", round(psnr_A - psnr_B, 4), "dB")
print("SSIM :", round(ssim_A - ssim_B, 4))

print()
print("=" * 75)
print("FILES CREATED")
print("=" * 75)

print("01_ground_truth_HR.png")
print("02_input_LR_x8.png")
print("03_bilinear_x2.png")
print("04_proposed_ASID_x4_final_x8.png")
print("05_direct_ASID_x8.png")

print()
print("Output directory:")
print(OUT_DIR)
print("=" * 75)
