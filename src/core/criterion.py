import torch.nn as nn
import torch
import torch.nn.functional as F


class MAELoss(nn.Module):
    def __init__(self, num_classes=10, scale=2.0):
        super(MAELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = (
            F.one_hot(labels.to(torch.int64), self.num_classes).float().to(pred.device)
        )
        loss = 1.0 - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * loss.mean()


class NMAE(nn.Module):
    def __init__(self, num_classes=10, scale=1.0):
        super(NMAE, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = (
            F.one_hot(labels.to(torch.int64), self.num_classes).float().to(pred.device)
        )
        norm = 1 / (self.num_classes - 1)
        loss = 1.0 - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * norm * loss.mean()
