#!/usr/bin/env python3

import os
import glob
import numpy as np
from PIL import Image

import torch

from tqdm import tqdm

from components.ASID_PreUpsample import ASID_PreUpsample
from utilities.utilities import calculate_psnr, calculate_ssim, tensor2img
from utilities.json_config import readConfig


CONFIG_PATH = "train_logs/ASID_PreUpsample_X8_DIV2K/model_config.json"
CHECKPOINT_CANDIDATES = [
    "checkpoints/epoch20_ASID.pth",
    "train_logs/ASID_PreUpsample_X8_DIV2K/checkpoints/epoch20_ASID.pth",
    "../checkpoints/ASID_PreUpsample_X8_DIV2K/epoch20_ASID.pth",
    "../checkpoints/epoch20_ASID.pth"
]
CHECKPOINT_PATH = next((p for p in CHECKPOINT_CANDIDATES if os.path.exists(p)), CHECKPOINT_CANDIDATES[0])
DATA_ROOT = "benchmark"


def load_model(device):

    cfg = readConfig(CONFIG_PATH)

    print("Loading PreUpsample ASID model...")
    print("Config:", CONFIG_PATH)
    print("Checkpoint:", CHECKPOINT_PATH)

    model = ASID_PreUpsample(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=cfg["feature_num"],
        **cfg["module_params"]
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    print("✓ Loaded epoch 20 PreUpsample ASID model")
    print("✓ Architecture: INTERNAL 2x Bilinear → ASID → 4x PixelShuffle")
    print()

    return model


def patch_inference(model, lr):

    tile = 64
    tile_overlap = 24

    # 2x internal pre-upsampling × 4x PixelShuffle = 8x
    scale = 8

    b, c, h, w = lr.size()

    tile = min(tile, h, w)

    stride = tile - tile_overlap

    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

    E = torch.zeros(
        b,
        c,
        h * scale,
        w * scale
    ).type_as(lr)

    W = torch.zeros_like(E)

    for h_idx in h_idx_list:

        for w_idx in w_idx_list:

            in_patch = lr[
                ...,
                h_idx:h_idx + tile,
                w_idx:w_idx + tile
            ]

            # IMPORTANT:
            # No external bilinear interpolation.
            # ASID_PreUpsample performs the 2x bilinear internally.
            out_patch = model(in_patch)

            if isinstance(out_patch, list):
                out_patch = out_patch[-1]

            out_patch_mask = torch.ones_like(out_patch)

            E[
                ...,
                h_idx * scale:(h_idx + tile) * scale,
                w_idx * scale:(w_idx + tile) * scale
            ].add_(out_patch)

            W[
                ...,
                h_idx * scale:(h_idx + tile) * scale,
                w_idx * scale:(w_idx + tile) * scale
            ].add_(out_patch_mask)

    return E.div_(W)


def evaluate_dataset(model, dataset_name, device):

    hr_dir = os.path.join(
        DATA_ROOT,
        "HR",
        dataset_name,
        "x8"
    )

    lr_dir = os.path.join(
        DATA_ROOT,
        "LR",
        "LRBI",
        dataset_name,
        "x8"
    )

    hr_files = sorted(
        glob.glob(
            os.path.join(hr_dir, "*.png")
        )
    )

    if not hr_files:
        print(f"ERROR: No HR images found in {hr_dir}")
        return None

    print()
    print("=" * 60)
    print(f"Processing {dataset_name}")
    print("=" * 60)
    print(f"HR images: {len(hr_files)}")

    psnr_values = []
    ssim_values = []

    with torch.no_grad():

        for hr_path in tqdm(hr_files):

            base_name = os.path.basename(hr_path)

            if base_name.endswith("_HR_x8.png"):
                base_name = base_name[:-len("_HR_x8.png")]
            else:
                base_name = os.path.splitext(base_name)[0]

            lr_path = os.path.join(
                lr_dir,
                f"{base_name}_LRBI_x8.png"
            )

            if not os.path.exists(lr_path):
                print(f"WARNING: Missing LR image: {lr_path}")
                continue

            # HR
            hr_pil = Image.open(hr_path).convert("RGB")
            hr_np = np.array(hr_pil).astype(np.float32)

            # LR
            lr_pil = Image.open(lr_path).convert("RGB")
            lr_np = np.array(lr_pil).astype(np.float32)

            # Convert LR HWC -> CHW
            lr_tensor = torch.from_numpy(
                lr_np.transpose(2, 0, 1)
            )

            # Same normalization as repository evaluator:
            # [0,255] -> [0,1] -> [-1,1]
            lr_tensor = (
                lr_tensor / 255.0 - 0.5
            ) * 2.0

            lr_tensor = lr_tensor.unsqueeze(0).to(device)

            # Correct Run A:
            # LR -> internal 2x bilinear -> ASID -> 4x PixelShuffle
            output = patch_inference(
                model,
                lr_tensor
            )

            # Same conversion as repository evaluator
            output_np = tensor2img(
                output.cpu()
            )[0]

            # Match HR dimensions
            h_hr, w_hr = hr_np.shape[:2]

            output_np = output_np[
                :h_hr,
                :w_hr,
                :
            ]

            psnr = calculate_psnr(
                output_np,
                hr_np
            )

            ssim = calculate_ssim(
                output_np,
                hr_np
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim)

    if not psnr_values:
        print("ERROR: No valid image pairs were evaluated.")
        return None

    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))

    print()
    print(f"{dataset_name} Results")
    print("-" * 40)
    print(f"Images : {len(psnr_values)}")
    print(f"PSNR   : {avg_psnr:.4f} dB")
    print(f"SSIM   : {avg_ssim:.4f}")

    return avg_psnr, avg_ssim


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("ASID PRE-UPsample ×8 EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(
        "Pipeline: LR → internal 2× Bilinear → ASID → 4× PixelShuffle"
    )
    print("Total scale: ×8")
    print("Checkpoint: epoch20_ASID.pth")
    print("Patch inference: 64×64, overlap 24")
    print("=" * 70)

    model = load_model(device)

    datasets = [
        "Set5",
        "Set14",
        "B100",
        "Urban100"
    ]

    results = {}

    for dataset_name in datasets:

        result = evaluate_dataset(
            model,
            dataset_name,
            device
        )

        if result is not None:
            results[dataset_name] = result

    print()
    print("=" * 70)
    print("FINAL PRE-UPsample ASID ×8 RESULTS")
    print("=" * 70)

    print(
        f"{'Dataset':<15}"
        f"{'PSNR (dB)':>15}"
        f"{'SSIM':>15}"
    )

    print("-" * 70)

    for dataset_name, (psnr, ssim) in results.items():

        print(
            f"{dataset_name:<15}"
            f"{psnr:>15.4f}"
            f"{ssim:>15.4f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
