import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits


class DistributionBalancedLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        pos_counts=None,
        neg_counts=None,
    ):
        super().__init__()

        self.reduction = reduction
        self.cls_criterion = binary_cross_entropy_with_logits

        # focal loss params
        self.gamma = 2.0
        self.balance_param = 2.0

        # mapping function params
        self.map_alpha = 0.1
        self.map_beta = 10.0
        self.map_gamma = 0.2

        self.pos_count = torch.from_numpy(pos_counts).float()
        self.neg_count = torch.from_numpy(neg_counts).float()
        self.num_classes = self.pos_count.shape[0]
        self.train_num = self.pos_count[0] + self.neg_count[0]

        # regularization params
        self.neg_scale = 2.0
        init_bias = 0.05

        self.init_bias = (
            -torch.log(self.train_num / self.pos_count - 1) * init_bias / self.neg_scale
        )

        self.freq_inv = torch.ones(self.pos_count.shape) / self.pos_count

    def forward(self, cls_score, label):
        cls_score = cls_score.clone()
        weight = self.rebalance_weight(label.float())
        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        # focal
        logpt = -self.cls_criterion(
            cls_score.clone(), label, weight=None, reduction="none"
        )
        # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
        pt = torch.exp(logpt)
        loss = self.cls_criterion(
            cls_score, label.float(), weight=weight, reduction="none"
        )
        loss = ((1 - pt) ** self.gamma) * loss
        loss = self.balance_param * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            self.reduction = loss.sum()

        return loss

    def logit_reg_functions(self, labels, logits, weight=None):
        self.init_bias = self.init_bias.to(logits)
        logits += self.init_bias
        logits = logits * (1 - labels) * self.neg_scale + logits * labels
        weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        self.freq_inv = self.freq_inv.to(gt_labels)
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = (
            torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma))
            + self.map_alpha
        )
        return weight


