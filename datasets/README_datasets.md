# Dataset Download Instructions

## Option 1: NIH ChestX-ray14 (Recommended)

**Size:** ~45 GB (112,120 images)  
**URL:** https://nihcc.app.box.com/v/ChestXray-NIHCC

### Download Steps:

1. Visit the NIH Box link above
2. Download all 12 zip files: `images_001.tar.gz` through `images_012.tar.gz`
3. Download `Data_Entry_2017.csv` (labels file)
4. Extract all images to `./datasets/images/`

```
datasets/
├── images/
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ... (112,120 images)
└── Data_Entry_2017.csv
```

### Label format in CSV:

```
Image Index,Finding Labels,...
00000001_000.png,Cardiomegaly|Effusion,...
00000002_000.png,No Finding,...
```

---

## Option 2: CheXpert (Larger, Higher Quality)

**Size:** ~439 GB (224,316 images)  
**URL:** https://stanfordmlgroup.github.io/competitions/chexpert/

Requires registration (free for research).

---

## Option 3: Small Demo Dataset

For testing without downloading large datasets, use the 5 sample images in `assets/`:

```bash
python model/train.py \
    --data_dir ./assets \
    --labels_file ./assets/demo_labels.csv \
    --epochs 5 \
    --batch_size 2
```

Note: Training on 5 images is purely for code verification, not clinical use.

---

## Cardiac Conditions in NIH Dataset

| Condition | # Images | % of Total |
|---|---|---|
| No Finding | 60,361 | 53.8% |
| Infiltration | 19,894 | 17.7% |
| Effusion | 13,317 | 11.9% |
| Atelectasis | 11,559 | 10.3% |
| Nodule | 6,331 | 5.6% |
| Cardiomegaly | 2,776 | 2.5% |
| Consolidation | 4,667 | 4.2% |
| Pleural Thickening | 3,385 | 3.0% |
| Edema | 2,303 | 2.1% |
