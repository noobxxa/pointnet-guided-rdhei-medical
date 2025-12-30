import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import LesionSegDataset
from pointnet2.models.pointnet2_lesion_seg_model import get_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--train_list", type=str, default="data/train.txt")
    p.add_argument("--val_list", type=str, default="data/val.txt")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_dir", type=str, default="outputs/ckpts")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()

def estimate_class_weights(dataset: LesionSegDataset):
    counts = np.zeros(2, dtype=np.int64)
    for i in range(len(dataset)):
        _, y = dataset[i]
        y = y.numpy()
        counts[0] += (y == 0).sum()
        counts[1] += (y == 1).sum()
    total = counts.sum()
    freq = counts / max(total, 1)
    w = 1.0 / (freq + 1e-6)
    w = w / w.sum() * 2
    return torch.tensor(w, dtype=torch.float32), counts

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = LesionSegDataset(args.train_list, root_dir=args.data_root, has_label=True, augment=True)
    val_ds = LesionSegDataset(args.val_list, root_dir=args.data_root, has_label=True, augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = get_model(num_classes=2, input_channels=3).to(device)

    class_w, counts = estimate_class_weights(train_ds)
    print("Class counts [0,1] =", counts.tolist())
    print("Class weights =", class_w.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))

    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_iou = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_dl:
            x = x.to(device)             # (B,3,N)
            y = y.to(device)             # (B,N)
            opt.zero_grad()
            logits = model(x)            # (B,2,N)
            logits = logits.permute(0, 2, 1).contiguous()  # (B,N,2)
            loss = criterion(logits.view(-1, 2), y.view(-1))
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"[{ep}/{args.epochs}] train loss={loss_sum/max(len(train_dl),1):.4f}")

        # val lesion IoU
        model.eval()
        inter, uni = 0, 0
        with torch.inference_mode():
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                pred = model(x).argmax(dim=1)  # (B,N)
                p1 = (pred == 1)
                g1 = (y == 1)
                inter += (p1 & g1).sum().item()
                uni += (p1 | g1).sum().item()
        iou = inter / (uni + 1e-6)
        print(f"          val lesion IoU={iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "val_lesion_iou": iou}, ckpt_path)
            print(f"          saved best -> {ckpt_path}")

if __name__ == "__main__":
    main()
