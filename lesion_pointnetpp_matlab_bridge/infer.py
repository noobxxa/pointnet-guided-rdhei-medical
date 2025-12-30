import os
import argparse
import numpy as np
import torch

from pointnet2.models.pointnet2_lesion_seg_model import get_model

TARGET_N = 8192

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint .pth. Leave empty for random weights (IO test).")
    p.add_argument("--input_npz", type=str, required=True, help="Input .npz with xyz (8192,3)")
    p.add_argument("--out_dir", type=str, default="outputs/preds")
    p.add_argument("--out_bin", type=str, default="", help="Optional explicit output .bin path (overrides out_dir).")
    p.add_argument("--device", type=str, default="cpu")  # 你现在是 torch+cpu，默认 cpu 最稳
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    data = np.load(args.input_npz)
    xyz = data["xyz"].astype(np.float32)
    assert xyz.shape == (TARGET_N, 3), f"Expected (8192,3), got {xyz.shape}"

    x = torch.from_numpy(xyz.T).unsqueeze(0).to(device)  # (1,3,N)

    model = get_model(num_classes=2, input_channels=3).to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        logits = model(x)  # (1,2,N)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # (N,)

    base = os.path.splitext(os.path.basename(args.input_npz))[0]

    if args.out_bin:
        bin_path = args.out_bin
        os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        bin_path = os.path.join(args.out_dir, f"{base}_pred_label.bin")

    pred.tofile(bin_path)
    print(f"Saved: {bin_path}  len={pred.size}")

    # debug npz（可留可删）
    if args.out_bin:
        npz_path = os.path.join(os.path.dirname(bin_path), f"{base}_pred.npz")
    else:
        npz_path = os.path.join(args.out_dir, f"{base}_pred.npz")
    np.savez_compressed(npz_path, xyz=xyz, pred_label=pred)
    print(f"Saved: {npz_path}")

if __name__ == "__main__":
    main()
