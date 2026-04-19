# Cell 4: Metrics + losses
def dilate_mask(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)


def erode_mask(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    return 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=pad)


def boundary_band_mask(gt: torch.Tensor, inner_k: int = 3, outer_k: int = 9) -> torch.Tensor:
    outer = dilate_mask(gt, outer_k)
    inner = erode_mask(gt, inner_k)
    return (outer - inner).clamp(0.0, 1.0)


def interior_mask(gt: torch.Tensor, erode_k: int = 9) -> torch.Tensor:
    return erode_mask(gt, erode_k)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def bce_iou_loss(logits: torch.Tensor, target: torch.Tensor):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    union = (prob + target - prob * target).sum(dim=(2, 3))
    iou_l = 1.0 - (inter + 1.0) / (union + 1.0)
    return bce + iou_l.mean(), {"bce": float(bce.detach()), "iou_loss": float(iou_l.mean().detach())}


def full_model_loss(outputs: Dict[str, torch.Tensor], target: torch.Tensor, cfg: TrainConfig):
    total, log_dict = bce_iou_loss(outputs["logits"], target)

    if "boundary_logits" in outputs:
        bgt = boundary_band_mask(target)
        b_logits = torch.nan_to_num(outputs["boundary_logits"], nan=0.0, posinf=20.0, neginf=-20.0)
        l_b = F.binary_cross_entropy_with_logits(b_logits, bgt)
        total = total + cfg.w_boundary * l_b
        log_dict["boundary_bce"] = float(l_b.detach())

    if "spectral_prior" in outputs and cfg.w_spectral > 0:
        # Optional (legacy) detached edge alignment loss. Keep disabled unless explicitly needed.
        pred_prob = torch.nan_to_num(torch.sigmoid(outputs["logits"]), nan=0.5, posinf=1.0, neginf=0.0)
        pred_edge = torch.nan_to_num(sobel_magnitude(pred_prob), nan=0.0, posinf=1.0, neginf=0.0)
        prior_det = torch.nan_to_num(outputs["spectral_prior"].detach(), nan=0.0, posinf=1.0, neginf=0.0)
        prior_det = prior_det.clamp(0.0, 1.0)
        l_s = F.l1_loss(pred_edge, prior_det)
        total = total + cfg.w_spectral * l_s
        log_dict["spectral_l1"] = float(l_s.detach())

    if "spectral_prior" in outputs and cfg.w_spec_consistency > 0:
        # Supervised spectral consistency (non-detached prior) against interior region
        prior = torch.nan_to_num(outputs["spectral_prior"], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if prior.shape[-2:] != target.shape[-2:]:
            prior = F.interpolate(prior, size=target.shape[-2:], mode="bilinear", align_corners=False)
        prior = prior.clamp(1e-4, 1.0 - 1e-4)
        gt_interior = erode_mask(target, k=5).clamp(1e-4, 1.0 - 1e-4)
        # BCE is unsafe under autocast; temporarily disable autocast for this op.
        with torch.amp.autocast(device_type=prior.device.type, enabled=False):
            l_sc = F.binary_cross_entropy(prior.float(), gt_interior.float())
        total = total + cfg.w_spec_consistency * l_sc
        log_dict["spec_consistency"] = float(l_sc.detach())

    if "spectral_discrepancy" in outputs and cfg.w_disc > 0:
        disc = torch.nan_to_num(outputs["spectral_discrepancy"], nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if disc.shape[-2:] != target.shape[-2:]:
            disc = F.interpolate(disc, size=target.shape[-2:], mode="bilinear", align_corners=False)
        disc = disc.clamp(1e-4, 1.0 - 1e-4)
        gt_band = boundary_band_mask(target)
        # BCE is unsafe under autocast; temporarily disable autocast for this op.
        with torch.amp.autocast(device_type=disc.device.type, enabled=False):
            l_disc = F.binary_cross_entropy(disc.float(), gt_band.float())
        total = total + cfg.w_disc * l_disc
        log_dict["disc_loss"] = float(l_disc.detach())

    return total, log_dict


def sanitize_outputs(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean = {}
    for k, v in outputs.items():
        if not torch.is_floating_point(v):
            clean[k] = v
            continue
        if k == "logits":
            clean[k] = torch.nan_to_num(v, nan=0.0, posinf=20.0, neginf=-20.0)
        elif k in ("boundary_logits",):
            clean[k] = torch.nan_to_num(v, nan=0.0, posinf=20.0, neginf=-20.0)
        elif k in ("spectral_prior", "spectral_discrepancy", "phase_map"):
            clean[k] = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        else:
            clean[k] = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
    return clean


def sanitize_model_params_(model: nn.Module) -> bool:
    repaired = False
    for p in model.parameters():
        if p is None:
            continue
        if not torch.isfinite(p.data).all():
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
            repaired = True
    return repaired


def sanitize_grads_(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1.0, neginf=-1.0)


def _safe_div(a, b):
    return a / (b + 1e-8)


def calc_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def calc_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + 1e-8))


def calc_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter) / (denom + 1e-8))


def calc_max_f(pred: np.ndarray, gt: np.ndarray, beta2: float = 0.3) -> float:
    gt_bin = gt >= 0.5
    max_f = 0.0
    for thr in np.linspace(0.0, 1.0, 21):
        pb = pred >= thr
        tp = np.logical_and(pb, gt_bin).sum()
        p = _safe_div(tp, pb.sum())
        r = _safe_div(tp, gt_bin.sum())
        f = _safe_div((1 + beta2) * p * r, (beta2 * p + r))
        max_f = max(max_f, float(f))
    return max_f


def _object_score(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    mu = x.mean()
    sigma = x.std()
    return float((2.0 * mu) / (mu * mu + 1.0 + sigma + 1e-8))


def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    x = pred.mean()
    y = gt.mean()
    if pred.size == 1:
        sigma_x2 = 0.0
        sigma_y2 = 0.0
        sigma_xy = 0.0
    else:
        sigma_x2 = pred.var(ddof=1)
        sigma_y2 = gt.var(ddof=1)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (pred.size - 1)
    alpha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)
    if alpha != 0:
        return float(alpha / (beta + 1e-8))
    if alpha == 0 and beta == 0:
        return 1.0
    return 0.0


def _centroid(gt: np.ndarray) -> Tuple[int, int]:
    h, w = gt.shape
    if gt.sum() == 0:
        return w // 2, h // 2
    ys, xs = np.where(gt > 0.5)
    x = int(np.round(xs.mean()))
    y = int(np.round(ys.mean()))
    return x, y


def _split_regions(pred: np.ndarray, gt: np.ndarray):
    h, w = gt.shape
    cx, cy = _centroid(gt)
    cx = np.clip(cx, 1, w - 1)
    cy = np.clip(cy, 1, h - 1)
    regions = [
        (slice(0, cy), slice(0, cx)),
        (slice(0, cy), slice(cx, w)),
        (slice(cy, h), slice(0, cx)),
        (slice(cy, h), slice(cx, w)),
    ]
    weights = []
    pred_parts = []
    gt_parts = []
    area = h * w
    for rs, cs in regions:
        p = pred[rs, cs]
        g = gt[rs, cs]
        pred_parts.append(p)
        gt_parts.append(g)
        weights.append(float((p.size) / (area + 1e-8)))
    return pred_parts, gt_parts, weights


def calc_s_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    pred = pred.astype(np.float32)
    gt = (gt >= 0.5).astype(np.float32)
    y = gt.mean()
    if y == 0:
        return float(1.0 - pred.mean())
    if y == 1:
        return float(pred.mean())

    fg = pred[gt == 1]
    bg = 1.0 - pred[gt == 0]
    so = y * _object_score(fg) + (1.0 - y) * _object_score(bg)

    pred_parts, gt_parts, weights = _split_regions(pred, gt)
    sr = 0.0
    for p, g, w in zip(pred_parts, gt_parts, weights):
        sr += w * _ssim(p, g)

    return float(alpha * so + (1.0 - alpha) * sr)


def calc_masked_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, mask_bin: np.ndarray) -> float:
    pred_m = np.logical_and(pred_bin, mask_bin)
    gt_m = np.logical_and(gt_bin, mask_bin)
    inter = np.logical_and(pred_m, gt_m).sum()
    union = np.logical_or(pred_m, gt_m).sum()
    if union == 0:
        # Undefined masked IoU for this sample; let downstream aggregation ignore it.
        return float("nan")
    return float(inter / (union + 1e-8))