def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


seed_everything(CFG.seed)


def list_images(img_dir: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    p = Path(img_dir)
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
    files.sort()
    return files


def find_gt_by_stem(gt_dir: str, stem: str) -> Path:
    p = Path(gt_dir)
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        cand = p / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def extract_category_cod10k(name: str) -> str:
    parts = name.split("-")
    # COD10K-CAM-{BG|NBI}-{Class}-{ID}
    if len(parts) < 5:
        return "Unknown"
    if parts[0] != "COD10K" or parts[1] != "CAM":
        return "Unknown"
    if parts[-1].isdigit():
        class_parts = parts[3:-1]
    else:
        class_parts = parts[3:]
    return "-".join(class_parts) if class_parts else "Unknown"


def extract_category_camo(name: str) -> str:
    # CAMO naming can vary; keep robust fallback
    if "_" in name:
        return name.split("_")[0]
    if "-" in name:
        return name.split("-")[0]
    return "Unknown"


def build_train_records(paths: DataPaths) -> List[Dict]:
    records = []
    for dataset_name, img_dir, gt_dir, category_fn in [
        ("COD10K", paths.train_img, paths.train_gt, extract_category_cod10k),
        ("CAMO", paths.camo_train_img, paths.camo_train_gt, extract_category_camo),
    ]:
        img_files = list_images(img_dir)
        missing = 0
        for img_path in img_files:
            gt_path = find_gt_by_stem(gt_dir, img_path.stem)
            if gt_path is None:
                missing += 1
                continue
            records.append(
                {
                    "id": f"{dataset_name}:{img_path.stem}",
                    "dataset": dataset_name,
                    "name": img_path.stem,
                    "category": category_fn(img_path.stem),
                    "img_path": str(img_path),
                    "gt_path": str(gt_path),
                }
            )
        print(f"{dataset_name} train pairs: {len(img_files) - missing} (missing gt: {missing})")

    print("Combined train pairs:", len(records))
    return records


def stratified_split(records: List[Dict], val_ratio: float, seed: int):
    groups = defaultdict(list)
    for idx, r in enumerate(records):
        key = f"{r['dataset']}::{r['category']}"
        groups[key].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for key, idxs in groups.items():
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train_idx.extend(idxs)
            continue
        k = max(1, int(round(n * val_ratio)))
        if k >= n:
            k = n - 1
        val_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    # Distribution check
    train_c = Counter((r["dataset"], r["category"]) for r in train_records)
    val_c = Counter((r["dataset"], r["category"]) for r in val_records)
    missing_in_val = [k for k in train_c.keys() if k not in val_c]
    if missing_in_val:
        print(f"Warning: {len(missing_in_val)} strata absent in val_tune")
    else:
        print("All train strata represented in val_tune")

    return train_records, val_records


def create_or_load_splits(paths: DataPaths, cfg: TrainConfig):
    split_dir = Path(cfg.save_root) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_file = split_dir / "train_val_split.json"

    if split_file.exists():
        data = json.loads(split_file.read_text())
        train_records = data["train_records"]
        val_records = data["val_records"]
        print("Loaded existing split:", split_file)
    else:
        records = build_train_records(paths)
        train_records, val_records = stratified_split(records, cfg.val_ratio, cfg.split_seed)
        split_obj = {
            "cfg": {"val_ratio": cfg.val_ratio, "split_seed": cfg.split_seed},
            "train_records": train_records,
            "val_records": val_records,
        }
        split_file.write_text(json.dumps(split_obj, indent=2))
        print("Created split:", split_file)

    print("train_tune:", len(train_records), "val_tune:", len(val_records))
    return train_records, val_records


train_records, val_records = create_or_load_splits(PATHS, CFG)