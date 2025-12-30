import os
import argparse
import numpy as np

TARGET_N = 8192

def load_xyz_file(path: str) -> np.ndarray:
    data = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = map(float, parts[:3])
            except ValueError:
                continue
            data.append([x, y, z])
    xyz = np.asarray(data, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Bad xyz shape from {path}: {xyz.shape}")
    return xyz

def fix_num_points(xyz: np.ndarray, target_n: int = TARGET_N, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = xyz.shape[0]
    if n == target_n:
        idx = np.arange(n)
    elif n > target_n:
        idx = rng.choice(n, size=target_n, replace=False)
    else:
        idx = rng.choice(n, size=target_n, replace=True)
    return xyz[idx], idx

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xyz", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    xyz = load_xyz_file(args.xyz)
    xyz_fixed, _ = fix_num_points(xyz, TARGET_N, seed=args.seed)

    # save npz (for PointNet++)
    out_npz = os.path.join(args.out_dir, "demo_case.npz")
    np.savez_compressed(out_npz, xyz=xyz_fixed)
    print(f"Saved: {out_npz}  xyz.shape={xyz_fixed.shape}")

    # save xyz (for MATLAB carrier)
    out_xyz = os.path.join(args.out_dir, "demo_case_8192.xyz")
    with open(out_xyz, "w") as f:
        for x, y, z in xyz_fixed:
            f.write(f"{x} {y} {z}\n")
    print(f"Saved: {out_xyz}  xyz.shape={xyz_fixed.shape}")

if __name__ == "__main__":
    main()
