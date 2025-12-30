import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- utils: pairwise distance / knn / gather ----------
def square_distance(src, dst):
    # src: (B, N, 3), dst: (B, M, 3) -> (B, N, M)
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))  # (B,N,M)
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def index_points(points, idx):
    # points: (B, N, C); idx: (B, S) or (B, S, K) -> (B, S, C) or (B, S, K, C)
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def knn_point(k, xyz, new_xyz):
    # xyz: (B, N, 3), new_xyz: (B, S, 3) -> idx: (B, S, k)
    dist = square_distance(new_xyz, xyz)  # (B,S,N)
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
    return idx


def farthest_point_sample(xyz, npoint):
    # xyz: (B, N, 3) -> centroids: (B, npoint)
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids


# --------- PointNet++ blocks: SA and FP (pure torch) ----------
class SharedMLP1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv1d(channels[i], channels[i+1], 1, bias=False),
                nn.BatchNorm1d(channels[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SharedMLP2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv2d(channels[i], channels[i+1], 1, bias=False),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PointnetSAModuleKNN(nn.Module):
    """
    SA with FPS + kNN grouping (no ball query).
    Input:
      xyz: (B, 3, N)
      points: (B, C, N) or None
    Output:
      new_xyz: (B, 3, S)
      new_points: (B, C', S)
    """
    def __init__(self, npoint, k, mlp_channels, use_xyz=True):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.use_xyz = use_xyz
        self.mlp = SharedMLP2d(mlp_channels)

    def forward(self, xyz, points):
        B, _, N = xyz.shape
        xyz_t = xyz.transpose(1, 2).contiguous()  # (B,N,3)

        # FPS -> (B,S)
        fps_idx = farthest_point_sample(xyz_t, self.npoint)
        new_xyz_t = index_points(xyz_t, fps_idx)          # (B,S,3)
        new_xyz = new_xyz_t.transpose(1, 2).contiguous()  # (B,3,S)

        # kNN grouping around new_xyz
        idx = knn_point(self.k, xyz_t, new_xyz_t)         # (B,S,k)
        grouped_xyz = index_points(xyz_t, idx)            # (B,S,k,3)
        grouped_xyz_norm = grouped_xyz - new_xyz_t.unsqueeze(2)

        if points is not None:
            points_t = points.transpose(1, 2).contiguous()     # (B,N,C)
            grouped_points = index_points(points_t, idx)       # (B,S,k,C)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B,S,k,3+C)
            else:
                new_features = grouped_points
        else:
            new_features = grouped_xyz_norm  # (B,S,k,3)

        # (B,S,k,C) -> (B,C,S,k)
        new_features = new_features.permute(0, 3, 1, 2).contiguous()
        new_features = self.mlp(new_features)  # (B,C',S,k)

        # max pool over k -> (B,C',S)
        new_points = torch.max(new_features, dim=-1)[0]
        return new_xyz, new_points


class PointnetFPModule(nn.Module):
    """
    Feature Propagation with 3-NN interpolation (pure torch).
    Input:
      xyz1: (B,3,N1) target (dense)
      xyz2: (B,3,N2) source (sparser)
      points1: (B,C1,N1) or None
      points2: (B,C2,N2)
    Output:
      new_points: (B,C',N1)
    """
    def __init__(self, mlp_channels):
        super().__init__()
        self.mlp = SharedMLP1d(mlp_channels)

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1_t = xyz1.transpose(1, 2).contiguous()  # (B,N1,3)
        xyz2_t = xyz2.transpose(1, 2).contiguous()  # (B,N2,3)

        dist = square_distance(xyz1_t, xyz2_t)      # (B,N1,N2)
        dist, idx = torch.topk(dist, k=3, dim=-1, largest=False, sorted=False)  # (B,N1,3)

        dist = torch.clamp(dist, min=1e-10)
        weight = 1.0 / dist
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (B,N1,3)

        points2_t = points2.transpose(1, 2).contiguous()  # (B,N2,C2)
        interpolated = torch.sum(index_points(points2_t, idx) * weight.unsqueeze(-1), dim=2)  # (B,N1,C2)
        interpolated = interpolated.transpose(1, 2).contiguous()  # (B,C2,N1)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=1)  # (B,C1+C2,N1)
        else:
            new_points = interpolated

        new_points = self.mlp(new_points)
        return new_points


class PointNet2PureSemSeg(nn.Module):
    """
    Pure torch PointNet++-style semseg using KNN SA + FP.
    Input:  (B,3,8192)
    Output: (B,2,8192)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # SA: progressively downsample
        self.sa1 = PointnetSAModuleKNN(npoint=2048, k=32, mlp_channels=[3, 64, 64, 128], use_xyz=True)
        self.sa2 = PointnetSAModuleKNN(npoint=512,  k=32, mlp_channels=[128+3, 128, 128, 256], use_xyz=True)
        self.sa3 = PointnetSAModuleKNN(npoint=128,  k=32, mlp_channels=[256+3, 256, 256, 512], use_xyz=True)

        # FP
        self.fp3 = PointnetFPModule(mlp_channels=[512 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp_channels=[256 + 128, 256, 128])
        self.fp1 = PointnetFPModule(mlp_channels=[128 + 0, 128, 128, 128])

        # classifier
        self.cls = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz):
        l0_xyz = xyz
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)     # (B,3,2048), (B,128,2048)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)     # (B,3,512),  (B,256,512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)     # (B,3,128),  (B,512,128)

        l2_points_fp = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B,256,512)
        l1_points_fp = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_fp)  # (B,128,2048)
        l0_points_fp = self.fp1(l0_xyz, l1_xyz, None, l1_points_fp)    # (B,128,8192)

        logits = self.cls(l0_points_fp)  # (B,2,8192)
        return logits
