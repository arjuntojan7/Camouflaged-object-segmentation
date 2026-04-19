! pip install pysodmetrics

# Cell: Standard COD evaluation with pysodmetrics
# NOTE: pysodmetrics (already installed) provides the py_sod_metrics namespace
# Do NOT re-install py_sod_metrics — it doesn't exist on PyPI

import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import py_sod_metrics

# ----------------------------
# Config
# ----------------------------
# Set which checkpoint and which model class to evaluate
EVAL_CKPT   = "/kaggle/working/sc_codnet_runs/B1_baseline/best_sm.pth"
EVAL_MODEL  = "B1"   # one of: "B0", "B1", "B2", "B2ESC"

SAVE_PRED   = False
PRED_ROOT   = Path("/kaggle/working/eval_preds")
OUT_JSON    = Path(CFG.save_root) / f"{EVAL_MODEL}_pysodmetrics_eval.json"

assert Path(EVAL_CKPT).exists(), f"Checkpoint not found: {EVAL_CKPT}"
assert EVAL_MODEL in ("B0", "B1", "B2", "B2ESC"), f"Unknown EVAL_MODEL: {EVAL_MODEL}"

# ----------------------------
# Model loader — always uses the correct class, strict=True
# ----------------------------
def load_model_for_eval_strict(model_type: str, ckpt_path: str) -> torch.nn.Module:
    """
    Load checkpoint into the correct model class with strict=True.
    This ensures you never accidentally evaluate a randomly-initialized module.
    """
    if model_type == "B0":
        model = B0Net()
    elif model_type == "B1":
        model = B1Net(tau=0.03, k_eig=8)
    elif model_type == "B2":
        model = B2Net(tau=0.03, k_eig=8)
    elif model_type == "B2ESC":
        model = B2ESCNet(tau=0.03, k_eig=8)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Strict load — will raise if there's a mismatch between checkpoint and model class
    missing, unexpected = model.load_state_dict(state, strict=True)

    # These should both be empty; if not, the wrong checkpoint/model pair was given
    if missing:
        raise RuntimeError(
            f"strict=True load failed for {model_type}.\n"
            f"Missing keys ({len(missing)}): {missing[:5]}...\n"
            f"Check that EVAL_CKPT was saved from a {model_type} training run."
        )
    if unexpected:
        raise RuntimeError(
            f"strict=True load failed for {model_type}.\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...\n"
            f"Check that EVAL_CKPT was saved from a {model_type} training run."
        )

    model.eval()
    print(f"Loaded {model_type} checkpoint: {ckpt_path}")

    # Print checkpoint metadata if available
    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch", "unknown")
        val_sm = ckpt.get("val_summary", {}).get("Sm", "unknown")
        print(f"  Checkpoint epoch={epoch}, saved val_Sm={val_sm}")

    return model


# ----------------------------
# Inference helper
# ----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_eval_norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


@torch.no_grad()
def infer_prob_map(model, pil_img: Image.Image, out_hw: tuple) -> np.ndarray:
    """
    Run model inference and return a probability map resized to out_hw (H, W).
    out_hw should match the original GT resolution so metrics are computed
    at the correct scale.
    """
    x = transforms.functional.resize(
        pil_img.convert("RGB"),
        [CFG.img_size, CFG.img_size],
        interpolation=InterpolationMode.BILINEAR,
    )
    x = transforms.functional.to_tensor(x)
    x = _eval_norm(x).unsqueeze(0).to(DEVICE, non_blocking=True)

    logits = model(x)["logits"]              # [1, 1, H', W']
    prob   = torch.sigmoid(logits)
    prob   = F.interpolate(
        prob,
        size=out_hw,                         # resize back to original GT size
        mode="bilinear",
        align_corners=False,
    )
    return prob[0, 0].clamp(0.0, 1.0).cpu().numpy()


# ----------------------------
# Dataset pair builder
# ----------------------------
def build_eval_pairs(img_dir: str, gt_dir: str):
    """Return sorted list of (img_path, gt_path) tuples with matched stems."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_files = sorted(
        f for f in Path(img_dir).iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )
    pairs = []
    missing = 0
    for ip in img_files:
        gp = find_gt_by_stem(gt_dir, ip.stem)
        if gp is None:
            missing += 1
            continue
        pairs.append((ip, gp))

    if missing > 0:
        print(f"  Warning: {missing} images had no matching GT and were skipped.")
    return pairs


# ----------------------------
# Per-dataset evaluation
# ----------------------------
def evaluate_dataset_pysod(
    model,
    dataset_name: str,
    img_dir: str,
    gt_dir: str,
    save_pred: bool = False,
) -> dict:
    pairs = build_eval_pairs(img_dir, gt_dir)
    if len(pairs) == 0:
        print(f"[{dataset_name}] no valid image/gt pairs found — skipping.")
        return None

    print(f"[{dataset_name}] evaluating {len(pairs)} images...")

    # Initialize metric accumulators
    fm  = py_sod_metrics.Fmeasure()
    wfm = py_sod_metrics.WeightedFmeasure()
    sm  = py_sod_metrics.Smeasure()
    em  = py_sod_metrics.Emeasure()
    mae = py_sod_metrics.MAE()

    if save_pred:
        pred_dir = PRED_ROOT / dataset_name
        pred_dir.mkdir(parents=True, exist_ok=True)

    for img_path, gt_path in tqdm(pairs, desc=f"  {dataset_name}", ncols=90):
        # Load inputs
        img    = Image.open(img_path).convert("RGB")
        gt     = Image.open(gt_path).convert("L")
        gt_np  = np.array(gt, dtype=np.uint8)

        # Standard GT binarization threshold used across all COD papers
        gt_bin = (gt_np > 127).astype(np.uint8) * 255

        # Run inference at original GT resolution
        pred_prob = infer_prob_map(model, img, out_hw=gt_bin.shape)
        pred_u8   = (pred_prob * 255.0).astype(np.uint8)

        if save_pred:
            Image.fromarray(pred_u8).save(
                PRED_ROOT / dataset_name / f"{img_path.stem}.png"
            )

        # Accumulate metrics
        fm.step(pred=pred_u8,  gt=gt_bin)
        wfm.step(pred=pred_u8, gt=gt_bin)
        sm.step(pred=pred_u8,  gt=gt_bin)
        em.step(pred=pred_u8,  gt=gt_bin)
        mae.step(pred=pred_u8, gt=gt_bin)

    # Collect results
    fm_res = fm.get_results()["fm"]    # {"adp": float, "curve": array}
    em_res = em.get_results()["em"]    # {"adp": float, "curve": array}

    result = {
        # Primary metrics — what papers report
        "Smeasure":  float(sm.get_results()["sm"]),
        "wFmeasure": float(wfm.get_results()["wfm"]),
        "MAE":       float(mae.get_results()["mae"]),
        # E-measure variants
        "adpEm":  float(em_res["adp"]),
        "meanEm": float(np.mean(em_res["curve"])),
        "maxEm":  float(np.max(em_res["curve"])),
        # F-measure variants
        "adpFm":  float(fm_res["adp"]),
        "meanFm": float(np.mean(fm_res["curve"])),
        "maxFm":  float(np.max(fm_res["curve"])),
        # Bookkeeping
        "num_images": int(len(pairs)),
        "model":      EVAL_MODEL,
        "ckpt":       str(EVAL_CKPT),
    }
    return result


# ----------------------------
# Run evaluation
# ----------------------------
eval_model = load_model_for_eval_strict(EVAL_MODEL, EVAL_CKPT)

eval_results = {}

for ds_name in ["COD10K", "CAMO", "CHAMELEON", "NC4K"]:
    if ds_name not in PATHS.test_sets:
        print(f"[{ds_name}] not found in PATHS.test_sets — skipping.")
        continue

    img_dir = PATHS.test_sets[ds_name]["img"]
    gt_dir  = PATHS.test_sets[ds_name]["gt"]

    if not Path(img_dir).exists():
        print(f"[{ds_name}] image dir missing: {img_dir}")
        continue
    if not Path(gt_dir).exists():
        print(f"[{ds_name}] GT dir missing: {gt_dir}")
        continue

    res = evaluate_dataset_pysod(
        model=eval_model,
        dataset_name=ds_name,
        img_dir=img_dir,
        gt_dir=gt_dir,
        save_pred=SAVE_PRED,
    )

    if res is not None:
        eval_results[ds_name] = res
        print(
            f"  Smeasure={res['Smeasure']:.4f}  "
            f"wFmeasure={res['wFmeasure']:.4f}  "
            f"MAE={res['MAE']:.4f}  "
            f"maxEm={res['maxEm']:.4f}  "
            f"maxFm={res['maxFm']:.4f}"
        )

# ----------------------------
# Save results
# ----------------------------
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(eval_results, indent=2))
print(f"\nSaved evaluation report -> {OUT_JSON}")