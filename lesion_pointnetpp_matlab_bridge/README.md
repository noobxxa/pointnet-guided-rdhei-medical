# Lesion vs Non-lesion Point Cloud Segmentation (PyTorch) → MATLAB label bridge

This repo is a **from-zero** runnable skeleton:
- Fixed point count **N=8192**
- Exports label as **uint8 binary** (`*_pred_label.bin`) for MATLAB (shape N×1)
- Model file is a **placeholder** MLP that matches the PointNet++ interface:
  - input: (B, 3, 8192)
  - output logits: (B, 2, 8192)
Replace `pointnet2/models/pointnet2_lesion_seg_model.py` with a real PointNet++ later.

## Quick start

### 1) Install deps (example)
Create a venv/conda env and install:
- python>=3.9
- numpy
- torch

### 2) Put your xyz file
Place your `.xyz` at:
`data/demo.xyz`

Expected per-line format:
`x y z [anything ...]` (only first 3 columns are used)

### 3) Preprocess to fixed 8192
```bash
python preprocess_xyz_to_npz.py
```
This creates:
`data/samples/demo_case.npz` containing `xyz (8192,3)`.

### 4) Run inference (no training required for IO test)
```bash
python infer.py --ckpt "" --input_npz data/samples/demo_case.npz --out_dir outputs/preds
```
If `--ckpt` is empty, it will run with random weights (for pipeline test only).

Output:
- `outputs/preds/demo_case_pred_label.bin` (uint8, length 8192)
- `outputs/preds/demo_case_pred.npz` (debug)

### 5) MATLAB read example
```matlab
fid = fopen('outputs/preds/demo_case_pred_label.bin','rb');
y = fread(fid, inf, 'uint8=>uint8'); fclose(fid);
assert(numel(y)==8192);
y = logical(y); % N×1
```

## Next step
When you have labeled training data (npz with `xyz` and `label`):
- create `data/train.txt`, `data/val.txt`
- run `python train.py ...`
- then infer with `--ckpt outputs/ckpts/best_model.pth`
