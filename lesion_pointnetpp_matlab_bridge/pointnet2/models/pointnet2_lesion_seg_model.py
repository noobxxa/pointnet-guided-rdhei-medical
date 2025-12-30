import torch.nn as nn
from .pointnet2_pure_torch import PointNet2PureSemSeg
def get_model(num_classes: int = 2, input_channels: int = 3) -> nn.Module:
    assert input_channels == 3, "This model expects xyz only (3 channels)."
    return PointNet2PureSemSeg(num_classes=num_classes)
