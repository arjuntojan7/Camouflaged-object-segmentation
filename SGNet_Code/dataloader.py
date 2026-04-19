from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import random
from typing import List, Dict


class CombinedTrainDataset(Dataset):
    def __init__(self, records: List[Dict], img_size: int = 384, augment: bool = False):
        self.records = records
        self.img_size = img_size
        self.augment = augment
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.records)

    def _augment(self, img, gt):
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            gt = transforms.functional.hflip(gt)
        if random.random() < 0.2:
            img = transforms.functional.vflip(img)
            gt = transforms.functional.vflip(gt)
        if random.random() < 0.2:
            angle = random.choice([90, 180, 270])
            img = transforms.functional.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            gt = transforms.functional.rotate(gt, angle, interpolation=InterpolationMode.NEAREST)
        if random.random() < 0.3:
            jitter = transforms.ColorJitter(0.15, 0.15, 0.10, 0.05)
            img = jitter(img)
        return img, gt

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["img_path"]).convert("RGB")
        gt = Image.open(r["gt_path"]).convert("L")
        if self.augment:
            img, gt = self._augment(img, gt)
        img = transforms.functional.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
        gt = transforms.functional.resize(gt, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
        img_t = transforms.functional.to_tensor(img)
        img_t = self.norm(img_t)
        gt_t = transforms.functional.to_tensor(gt)
        gt_t = (gt_t > 0.5).float()
        return {"image": img_t, "mask": gt_t, "id": r["id"], "dataset": r["dataset"]}


def make_loader(records, img_size, batch_size, num_workers, augment=False, shuffle=False, drop_last=False):
    ds = CombinedTrainDataset(records, img_size=img_size, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, drop_last=drop_last)
