# COD Camouflage Object Detection — B1Net

A PyTorch implementation of **B1Net**: a camouflage object detection model built on the **SMT-Tiny** backbone with a **D4 Spectral Prior** and **FPN decoder**, trained on COD10K + CAMO datasets.

## Project Structure

| File | Description |
|------|-------------|
| `01_data_split.py` | Filter COD10K by filename pattern and create train/test split lists |
| `02_imports_config.py` | All imports, `DataPaths`, and `TrainConfig` dataclasses |
| `03_utils.py` | Seeding, file utilities, category extractors, split builders |
| `04_dataloader.py` | `CombinedTrainDataset` and `make_loader` |
| `05_metrics_loss.py` | Loss functions (BCE+IoU, boundary, spectral) and evaluation metrics (Sm, maxF, MAE, IoU) |
| `06_smt_backbone.py` | SMT-Tiny transformer backbone (Scale-Modulated Transformer) |
| `07_model_b1net.py` | `B1Net` = SMT + D4SpectralPrior + FPNDecoder |
| `08_train.py` | Training loop with AMP, grad clipping, LR scheduling |
| `09_evaluate.py` | Model loading and val-set evaluation helpers |
| `10_eval_pysod.py` | Standard COD evaluation using `pysodmetrics` (Smeasure, wFm, MAE, Em, Fm) |

## Datasets

- **Train**: COD10K (3040 images) + CAMO (1000 images) = 4040 total
- **Test**: COD10K (2026), CAMO (250), CHAMELEON (76), NC4K (4121)

## Requirements

```bash
pip install torch torchvision timm ptflops thop yacs pysodmetrics tqdm
```

## Usage (Kaggle)

Run cells in order:
1. `01_data_split.py` — generate split list `.txt` files
2. `02_imports_config.py` → `03_utils.py` → `04_dataloader.py` — setup
3. `06_smt_backbone.py` → `07_model_b1net.py` — define models
4. `05_metrics_loss.py` → `08_train.py` — train B1Net
5. `10_eval_pysod.py` — evaluate on test sets

## Model: B1Net

- **Backbone**: SMT-Tiny (Scale-Modulated Transformer)
- **Spectral module**: D4SpectralPrior (graph Laplacian eigenvector-based)
- **Decoder**: FPN (Feature Pyramid Network)
- **Losses**: BCE + IoU + boundary BCE + spectral consistency + spectral discrepancy

## Key Hyperparameters (TrainConfig)

| Param | Value |
|-------|-------|
| img_size | 384 |
| batch_size | 8 |
| epochs | 60 |
| lr | 3e-4 |
| backbone_lr_mult | 0.3 |
| backbone_warmup_epochs | 10 |
