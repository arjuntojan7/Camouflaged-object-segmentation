import os
import random

print("=" * 70)
print("FILTERING COD10K BY FILENAME PATTERN (CAM/NON/BG)")
print("=" * 70)

os.makedirs('/kaggle/working/split_lists', exist_ok=True)

def categorize_by_filename(img_dir):
    """Categorize images based on filename pattern"""
    all_files = sorted([f for f in os.listdir(img_dir) 
                       if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))])
    
    camouflaged = []
    non_camouflaged = []
    background = []
    unknown = []
    
    for filename in all_files:
        name_without_ext = os.path.splitext(filename)[0]
        
        if '-CAM-' in filename:
            camouflaged.append(name_without_ext)
        elif '-NON-' in filename or 'NON' in filename.upper():
            non_camouflaged.append(name_without_ext)
        elif '-BG-' in filename or 'BACKGROUND' in filename.upper():
            background.append(name_without_ext)
        else:
            unknown.append(name_without_ext)
    
    return camouflaged, non_camouflaged, background, unknown

# ==================== COD10K TRAIN ====================
print("\n📁 COD10K TRAIN FOLDER:")
train_dir = '/kaggle/input/datasets/boatshuai/cod10k-v3/COD10K-v3/Train/Image'

train_cam, train_non, train_bg, train_unk = categorize_by_filename(train_dir)

print(f"   ✅ Camouflaged (-CAM-):       {len(train_cam)} images")
print(f"   ❌ Non-camouflaged (-NON-):   {len(train_non)} images")
print(f"   ❌ Background (-BG-):         {len(train_bg)} images")
print(f"   ⚠️  Unknown pattern:          {len(train_unk)} images")

# ==================== COD10K TEST ====================
print("\n📁 COD10K TEST FOLDER:")
test_dir = '/kaggle/input/datasets/boatshuai/cod10k-v3/COD10K-v3/Test/Image'

test_cam, test_non, test_bg, test_unk = categorize_by_filename(test_dir)

print(f"   ✅ Camouflaged (-CAM-):       {len(test_cam)} images")
print(f"   ❌ Non-camouflaged (-NON-):   {len(test_non)} images")
print(f"   ❌ Background (-BG-):         {len(test_bg)} images")
print(f"   ⚠️  Unknown pattern:          {len(test_unk)} images")

# ==================== SELECT FOR TRAINING ====================
print("\n" + "=" * 70)
print("SELECTING IMAGES FOR OFFICIAL SPLITS")
print("=" * 70)

# For COD10K, select exactly 3040 train and 2026 test from camouflaged
random.seed(42)

if len(train_cam) >= 3040:
    train_selected = sorted(random.sample(train_cam, 3040))
    print(f"\n✅ COD10K Train: Selected 3040 from {len(train_cam)} camouflaged images")
else:
    train_selected = train_cam
    print(f"\n⚠️  COD10K Train: Only {len(train_cam)} camouflaged images (need 3040)")

if len(test_cam) >= 2026:
    test_selected = sorted(random.sample(test_cam, 2026))
    print(f"✅ COD10K Test: Selected 2026 from {len(test_cam)} camouflaged images")
else:
    test_selected = test_cam
    print(f"⚠️  COD10K Test: Only {len(test_cam)} camouflaged images (need 2026)")

# Save COD10K splits
with open('/kaggle/working/split_lists/COD10K_train.txt', 'w') as f:
    f.write('\n'.join(train_selected))

with open('/kaggle/working/split_lists/COD10K_test.txt', 'w') as f:
    f.write('\n'.join(test_selected))

# ==================== CAMO (unchanged) ====================
print("\n📁 CAMO Dataset:")
camo_train_dir = '/kaggle/input/datasets/ivanomelchenkoim11/camo-dataset/CAMO-V.1.0-CVIU2019/Images/Train'
camo_test_dir = '/kaggle/input/datasets/ivanomelchenkoim11/camo-dataset/CAMO-V.1.0-CVIU2019/Images/Test'

camo_train_files = sorted([os.path.splitext(f)[0] for f in os.listdir(camo_train_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))])

camo_test_files = sorted([os.path.splitext(f)[0] for f in os.listdir(camo_test_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))])

with open('/kaggle/working/split_lists/CAMO_train.txt', 'w') as f:
    f.write('\n'.join(camo_train_files))

with open('/kaggle/working/split_lists/CAMO_test.txt', 'w') as f:
    f.write('\n'.join(camo_test_files))

print(f"   ✅ Train: {len(camo_train_files)} images (all camouflaged)")
print(f"   ✅ Test: {len(camo_test_files)} images (all camouflaged)")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 70)
print("✅ SPLIT LISTS CREATED WITH CAMOUFLAGED IMAGES ONLY!")
print("=" * 70)
print(f"\nDataset Breakdown:")
print(f"  COD10K Train: {len(train_selected)} camouflaged images")
print(f"  COD10K Test:  {len(test_selected)} camouflaged images")
print(f"  CAMO Train:   {len(camo_train_files)} images")
print(f"  CAMO Test:    {len(camo_test_files)} images")
print(f"\n  Total Training: {len(train_selected) + len(camo_train_files)} (Expected: 4,040)")
print(f"  Total Testing:  {len(test_selected) + len(camo_test_files)} (Expected: 2,276)")

if (len(train_selected) + len(camo_train_files)) == 4040:
    print("\n🎉 PERFECT! Exactly 4,040 training images!")
elif (len(train_selected) + len(camo_train_files)) >= 3900:
    print("\n✅ EXCELLENT! Very close to target (97%+)")
else:
    print(f"\n⚠️  Only {len(train_selected) + len(camo_train_files)} training images")

print("\n📂 Split list files saved to:")
print("   /kaggle/working/split_lists/COD10K_train.txt")
print("   /kaggle/working/split_lists/COD10K_test.txt")
print("   /kaggle/working/split_lists/CAMO_train.txt")
print("   /kaggle/working/split_lists/CAMO_test.txt")

print("\n🎯 Now run the dataset splitting script to organize your datasets!")

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def load_split_list(split_file):
    """Load image names from split list file"""
    with open(split_file, 'r') as f:
        image_names = [line.strip() for line in f.readlines() if line.strip()]
    return image_names

def find_and_copy_file(filename, source_dirs, dest_dir, extensions=['.jpg', '.png', '.jpeg', '.JPG', '.PNG']):
    """Search for a file and copy to destination"""
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            continue
        for ext in extensions:
            file_path = os.path.join(source_dir, filename + ext)
            if os.path.exists(file_path):
                shutil.copy2(file_path, dest_dir)
                return True
    return False

def create_dataset_split(source_img_dirs, source_gt_dirs, 
                        output_img_dir, output_gt_dir,
                        split_list_file, dataset_name):
    """Create dataset split based on official list"""
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_gt_dir, exist_ok=True)
    
    image_names = load_split_list(split_list_file)
    print(f"\n{dataset_name}: Processing {len(image_names)} images...")
    
    img_copied = 0
    gt_copied = 0
    
    for img_name in tqdm(image_names, desc=f"{dataset_name}"):
        if find_and_copy_file(img_name, source_img_dirs, output_img_dir):
            img_copied += 1
        if find_and_copy_file(img_name, source_gt_dirs, output_gt_dir):
            gt_copied += 1
    
    print(f"  ✅ Images: {img_copied}/{len(image_names)}")
    print(f"  ✅ GT: {gt_copied}/{len(image_names)}")
    
    return img_copied, gt_copied

# ==================== MAIN EXECUTION ====================
OUTPUT_ROOT = '/kaggle/working/datasets/'

print("=" * 70)
print("ORGANIZING DATASETS WITH OFFICIAL SPLITS")
print("=" * 70)

# COD10K
print("\n📁 COD10K Dataset:")
cod_train_img, cod_train_gt = create_dataset_split(
    ['/kaggle/input/datasets/boatshuai/cod10k-v3/COD10K-v3/Train/Image'],
    ['/kaggle/input/datasets/boatshuai/cod10k-v3/COD10K-v3/Train/GT_Object'],
    f'{OUTPUT_ROOT}/COD10K/Train/Image',
    f'{OUTPUT_ROOT}/COD10K/Train/GT',
    '/kaggle/working/split_lists/COD10K_train.txt',
    'COD10K Train'
)

cod_test_img, cod_test_gt = create_dataset_split(
    ['/kaggle/input/datasets/boatshuai/cod10k-v3/COD10K-v3/Test/Image'],
    ['/kaggle/input/datasets/boatshuai/cod10k-v3/COD10K-v3/Test/GT_Object'],
    f'{OUTPUT_ROOT}/COD10K/Test/Image',
    f'{OUTPUT_ROOT}/COD10K/Test/GT',
    '/kaggle/working/split_lists/COD10K_test.txt',
    'COD10K Test'
)

# CAMO
print("\n📁 CAMO Dataset:")
camo_train_img, camo_train_gt = create_dataset_split(
    ['/kaggle/input/datasets/ivanomelchenkoim11/camo-dataset/CAMO-V.1.0-CVIU2019/Images/Train'],
    ['/kaggle/input/datasets/ivanomelchenkoim11/camo-dataset/CAMO-V.1.0-CVIU2019/GT'],
    f'{OUTPUT_ROOT}/CAMO/Train/Image',
    f'{OUTPUT_ROOT}/CAMO/Train/GT',
    '/kaggle/working/split_lists/CAMO_train.txt',
    'CAMO Train'
)

camo_test_img, camo_test_gt = create_dataset_split(
    ['/kaggle/input/datasets/ivanomelchenkoim11/camo-dataset/CAMO-V.1.0-CVIU2019/Images/Test'],
    ['/kaggle/input/datasets/ivanomelchenkoim11/camo-dataset/CAMO-V.1.0-CVIU2019/GT'],
    f'{OUTPUT_ROOT}/CAMO/Test/Image',
    f'{OUTPUT_ROOT}/CAMO/Test/GT',
    '/kaggle/working/split_lists/CAMO_test.txt',
    'CAMO Test'
)

# SUMMARY
print("\n" + "=" * 70)
print("✅ DATASET ORGANIZATION COMPLETE!")
print("=" * 70)
print(f"\nFinal Counts:")
print(f"  COD10K Train: {cod_train_img} images, {cod_train_gt} GT")
print(f"  COD10K Test:  {cod_test_img} images, {cod_test_gt} GT")
print(f"  CAMO Train:   {camo_train_img} images, {camo_train_gt} GT")
print(f"  CAMO Test:    {camo_test_img} images, {camo_test_gt} GT")
print(f"\n  Total Training: {cod_train_img + camo_train_img} (Expected: 4,040)")
print(f"  Total Testing:  {cod_test_img + camo_test_img} (Expected: 2,276)")

print(f"\n📁 Organized datasets location: {OUTPUT_ROOT}")
print("\n🎯 Update your training script to:")
print("   train_dirs = [")
print("       '/kaggle/working/datasets/COD10K/Train/',")
print("       '/kaggle/working/datasets/CAMO/Train/'")
print("   ]")

class CombinedTrainDataset(Dataset):
    def __init__(self, records: List[Dict], img_size: int = 384, augment: bool = False):
        self.records = records
        self.img_size = img_size
        self.augment = augment
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.records)

    def _augment(self, img: Image.Image, gt: Image.Image):
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

        img = transforms.functional.resize(
            img, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR
        )
        gt = transforms.functional.resize(
            gt, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST
        )

        img_t = transforms.functional.to_tensor(img)
        img_t = self.norm(img_t)
        gt_t = transforms.functional.to_tensor(gt)
        gt_t = (gt_t > 0.5).float()

        return {
            "image": img_t,
            "mask": gt_t,
            "id": r["id"],
            "dataset": r["dataset"],
        }


def make_loader(records, img_size, batch_size, num_workers, augment=False, shuffle=False, drop_last=False):
    ds = CombinedTrainDataset(records, img_size=img_size, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


train_loader = make_loader(
    train_records, CFG.img_size, CFG.batch_size, CFG.num_workers, augment=True, shuffle=True, drop_last=True
)
val_loader = make_loader(
    val_records, CFG.img_size, CFG.val_batch_size, CFG.num_workers, augment=False, shuffle=False, drop_last=False
)

print("Train batches:", len(train_loader), "Val batches:", len(val_loader))