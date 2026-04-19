import os
import json
import math
import pickle
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import InterpolationMode


@dataclass
class DataPaths:
    train_img: str = "/kaggle/working/datasets/COD10K/Train/Image"
    train_gt: str = "/kaggle/working/datasets/COD10K/Train/GT"
    camo_train_img: str = "/kaggle/working/datasets/CAMO/Train/Image"
    camo_train_gt: str = "/kaggle/working/datasets/CAMO/Train/GT"
    test_sets: Dict[str, Dict[str, str]] = None

    def __post_init__(self):
        if self.test_sets is None:
            self.test_sets = {
                "COD10K": {
                    "img": "/kaggle/working/datasets/COD10K/Test/Image",
                    "gt": "/kaggle/working/datasets/COD10K/Test/GT",
                },
                "CAMO": {
                    "img": "/kaggle/working/datasets/CAMO/Test/Image",
                    "gt": "/kaggle/working/datasets/CAMO/Test/GT",
                },
                "CHAMELEON": {
                    "img": "/kaggle/input/datasets/wxli408/cod-test/TestDataset/CHAMELEON/Imgs",
                    "gt": "/kaggle/input/datasets/wxli408/cod-test/TestDataset/CHAMELEON/GT",
                },
                "NC4K": {
                    "img": "/kaggle/input/datasets/poorvaahuja/nc4k-camouflage/Imgs",
                    "gt": "/kaggle/input/datasets/poorvaahuja/nc4k-camouflage/GT",
                },
            }


@dataclass
class TrainConfig:
    seed: int = 42
    img_size: int = 384
    batch_size: int = 8
    val_batch_size: int = 8
    num_workers: int = 4
    epochs: int = 60
    lr: float = 3e-4
    min_lr: float = 1e-6
    backbone_lr_mult: float = 0.3        # ← was 0.5 (too aggressive → hurt CHAMELEON)
    weight_decay: float = 2e-4           # ← was 1e-4 (stronger regularisation helps OOD)
    grad_clip: float = 1.0
    amp: bool = True
    backbone_warmup_epochs: int = 10
    val_ratio: float = 0.10
    split_seed: int = 42
    save_root: str = "/kaggle/working/sc_codnet_runs"
    smt_ckpt: str = "/kaggle/input/datasets/study7/smt-tiny/smt_tiny.pth"
    w_boundary: float = 0.2             # ← was 0.20 (dead code — B1Net never outputs boundary_logits)
    w_spectral: float = 0.0
    w_spec_consistency: float = 0.10
    w_disc: float = 0.05
    delta_sm_strong: float = 0.008
    delta_sm_moderate: float = 0.003
    bootstrap_iters: int = 2000

PATHS = DataPaths()
CFG = TrainConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CFG.save_root, exist_ok=True)
print("Device:", DEVICE)
print("Save root:", CFG.save_root)