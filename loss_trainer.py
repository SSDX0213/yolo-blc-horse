from ultralytics import YOLO
from ultralytics.models.yolo.segment.train import SegmentationTrainer
import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return bce + dice


class MySegTrainer(SegmentationTrainer):
    def build_loss(self):
        print(">>> Using Custom DiceBCE Loss <<<")
        return DiceBCELoss()
