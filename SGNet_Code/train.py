class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, x, n=1):
        self.sum += float(x) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def metrics_from_batch(prob: torch.Tensor, gt: torch.Tensor):
    # prob/gt: [B,1,H,W], on CPU
    out = []
    for i in range(prob.shape[0]):
        p = prob[i, 0].numpy().astype(np.float32)
        g = gt[i, 0].numpy().astype(np.float32)
        pb = p >= 0.5
        gb = g >= 0.5

        bmask = boundary_band_mask(torch.from_numpy(g[None, None]).float(), 3, 9)[0, 0].numpy() >= 0.5
        imask = interior_mask(torch.from_numpy(g[None, None]).float(), 9)[0, 0].numpy() >= 0.5

        out.append(
            {
                "Sm": calc_s_measure(p, g),
                "maxF": calc_max_f(p, g),
                "MAE": calc_mae(p, g),
                "IoU": calc_iou(pb, gb),
                "Dice": calc_dice(pb, gb),
                "IoU_band": calc_masked_iou(pb, gb, bmask),
                "IoU_interior": calc_masked_iou(pb, gb, imask),
            }
        )
    return out


@torch.no_grad()
def evaluate_model(model, loader, cfg: TrainConfig, model_type: str):
    model.eval()
    loss_m = AverageMeter()
    per_image = []

    for batch in loader:
        x = batch["image"].to(DEVICE, non_blocking=True)
        y = batch["mask"].to(DEVICE, non_blocking=True)
        outputs = sanitize_outputs(model(x))

        loss, _ = full_model_loss(outputs, y, cfg)
        loss_m.update(loss.item(), x.size(0))

        prob = torch.sigmoid(outputs["logits"]).detach().cpu()
        gt = y.detach().cpu()
        per_image.extend(metrics_from_batch(prob, gt))

    summary = {}
    keys = per_image[0].keys() if len(per_image) else []
    for k in keys:
        arr = np.asarray([m[k] for m in per_image], dtype=np.float32)
        finite = np.isfinite(arr)
        summary[k] = float(np.nanmean(arr)) if finite.any() else float("nan")
    summary["val_loss"] = loss_m.avg
    return summary, per_image


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    bb_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in n:
            bb_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": cfg.lr * cfg.backbone_lr_mult},
            {"params": head_params, "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    return optimizer


def train_experiment(
    model: nn.Module,
    model_type: str,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    run_name: str,
):
    run_dir = Path(cfg.save_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    warmup_epochs = int(getattr(cfg, "backbone_warmup_epochs", 0))
    if warmup_epochs > 0:
        for n, p in model.named_parameters():
            if "backbone" in n:
                p.requires_grad_(False)

    optimizer = build_optimizer(model, cfg)
    if warmup_epochs > 0:
        if hasattr(torch.optim.lr_scheduler, "ConstantLR"):
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=warmup_epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=cfg.amp and DEVICE == "cuda")

    best_metric = -1.0
    best_sm = -1.0
    best_ckpt = run_dir / "best.pth"
    best_sm_ckpt = run_dir / "best_sm.pth"
    history = []

    for epoch in range(1, cfg.epochs + 1):
        if warmup_epochs > 0 and epoch == warmup_epochs + 1:
            for n, p in model.named_parameters():
                if "backbone" in n:
                    p.requires_grad_(True)
            optimizer = build_optimizer(model, cfg)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(cfg.epochs - warmup_epochs, 1), eta_min=cfg.min_lr
            )
            print(f"[{run_name}] backbone unfrozen at epoch {epoch}")

        model.train()
        train_loss_m = AverageMeter()

        for batch in train_loader:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["mask"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE, enabled=cfg.amp and DEVICE == "cuda"):
                raw_outputs = model(x)
                if not torch.isfinite(raw_outputs["logits"]).all():
                    repaired = sanitize_model_params_(model)
                    if repaired:
                        raw_outputs = model(x)
                    else:
                        print(f"[{run_name}] non-finite logits (no params repaired), skipping batch")
                        continue
                outputs = sanitize_outputs(raw_outputs)
                if not torch.isfinite(outputs["logits"]).all():
                    print(f"[{run_name}] non-finite logits after sanitize, skipping batch")
                    continue
                loss, _ = full_model_loss(outputs, y, cfg)

            if not torch.isfinite(loss):
                print(f"[{run_name}] non-finite loss encountered, skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            sanitize_grads_(model)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss_m.update(loss.item(), x.size(0))

        scheduler.step()
        val_summary, _ = evaluate_model(model, val_loader, cfg, model_type=model_type)

        row = {
            "epoch": epoch,
            "train_loss": train_loss_m.avg,
            "val_loss": val_summary["val_loss"],
            "Sm": val_summary["Sm"],
            "maxF": val_summary["maxF"],
            "MAE": val_summary["MAE"],
            "IoU": val_summary["IoU"],
            "Dice": val_summary["Dice"],
            "IoU_interior": val_summary["IoU_interior"],
            "IoU_band": val_summary["IoU_band"],
        }
        history.append(row)
        print(
            f"[{run_name}] epoch {epoch:03d}/{cfg.epochs} "
            f"train_loss={row['train_loss']:.4f} val_Sm={row['Sm']:.4f} "
            f"val_IoU={row['IoU']:.4f} val_intIoU={row['IoU_interior']:.4f}"
        )

        current = 0.6 * row["IoU_interior"] + 0.4 * row["Sm"]
        if not np.isfinite(current):
            current = -1.0
        if current > best_metric:
            best_metric = current
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_summary": val_summary,
                    "cfg": asdict(cfg),
                    "model_type": model_type,
                },
                best_ckpt,
            )
        if row["Sm"] > best_sm:
            best_sm = row["Sm"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_summary": val_summary,
                    "cfg": asdict(cfg),
                    "model_type": model_type,
                },
                best_sm_ckpt,
            )

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"[{run_name}] best combo={best_metric:.4f}, ckpt={best_ckpt}")
    print(f"[{run_name}] best Sm={best_sm:.4f}, ckpt={best_sm_ckpt}")
    return str(best_ckpt), history