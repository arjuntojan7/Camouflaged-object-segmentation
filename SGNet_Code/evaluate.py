!pip install yacs

RUN_B1 = True
B1_RUN_NAME = "B1_baseline"

b1_ckpt = None

if RUN_B1:
    seed_everything(CFG.seed)
    b1_model = B1Net(tau=0.03, k_eig=8).to(DEVICE)
    load_smt_pretrained(b1_model.backbone, CFG.smt_ckpt)
    b1_ckpt, _ = train_experiment(
        model=b1_model,
        model_type="B1",
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=CFG,
        run_name=B1_RUN_NAME,
    )

def load_b1_for_eval(ckpt_path: str):
    model = B1Net(tau=0.03, k_eig=8).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


if b1_ckpt is not None and Path(str(b1_ckpt)).exists():
    b1_eval_model = load_b1_for_eval(b1_ckpt)
    b1_summary, _ = evaluate_model(b1_eval_model, val_loader, CFG, model_type="B1")
    print(f"[B1] val_Sm={b1_summary['Sm']:.5f} val_IoU={b1_summary['IoU']:.5f}")

# def build_test_records(img_dir: str, gt_dir: str, dataset_name: str) -> List[Dict]:
#     img_files = list_images(img_dir)
#     recs = []
#     for ip in img_files:
#         gp = find_gt_by_stem(gt_dir, ip.stem)
#         if gp is None:
#             continue
#         recs.append(
#             {
#                 "id": f"{dataset_name}:{ip.stem}",
#                 "dataset": dataset_name,
#                 "name": ip.stem,
#                 "img_path": str(ip),
#                 "gt_path": str(gp),
#             }
#         )
#     print(f"{dataset_name}: {len(recs)} pairs")
#     return recs


# def evaluate_on_named_set(model, model_type: str, records: List[Dict], cfg: TrainConfig):
#     loader = make_loader(
#         records=records,
#         img_size=cfg.img_size,
#         batch_size=cfg.val_batch_size,
#         num_workers=cfg.num_workers,
#         augment=False,
#         shuffle=False,
#     )
#     summary, _ = evaluate_model(model, loader, cfg, model_type=model_type)
#     return summary


# EVAL_TESTS = True
# if EVAL_TESTS:
#     if b1_ckpt is None or (not Path(str(b1_ckpt)).exists()):
#         print("B1 checkpoint missing. Run Cell 8 first.")
#     else:
#         b1_eval_model = load_b1_for_eval(b1_ckpt)

#         test_report = {"B1": {}}
#         for ds_name, ds_paths in PATHS.test_sets.items():
#             if not Path(ds_paths["img"]).exists() or not Path(ds_paths["gt"]).exists():
#                 print(f"Skip {ds_name}: path missing")
#                 continue
#             recs = build_test_records(ds_paths["img"], ds_paths["gt"], ds_name)
#             if len(recs) == 0:
#                 print(f"Skip {ds_name}: no valid pairs")
#                 continue

#             b1_sum = evaluate_on_named_set(b1_eval_model, "B1", recs, CFG)
#             test_report["B1"][ds_name] = b1_sum

#             print(f"\n[{ds_name}]")
#             print(f"  B1   Sm={b1_sum['Sm']:.5f} IoU={b1_sum['IoU']:.5f} maxF={b1_sum['maxF']:.5f}")

#         tpath = Path(CFG.save_root) / "test_eval_report.json"
#         tpath.write_text(json.dumps(test_report, indent=2))
#         print("\nSaved test evaluation report:", tpath)