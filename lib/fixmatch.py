from typing import Dict

import torch
from torch.nn import functional as F

from lib.core import AUModel


class FixMatchModel(AUModel):
    STEP_OUTPUT = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]

    def __init__(self, p_cutoff: float = 0.95, lambda_u: float = 1., **kwargs):
        super().__init__(**kwargs)

        self.p_cutoff = p_cutoff
        self.lambda_u = lambda_u

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.backbone(x)[-1]
        outs = {task: head(x).squeeze() for task, head in self.heads.items()}
        return outs

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x_lb = batch["labeled"]["img"]
        y_lb = batch["labeled"]["multilabel"]

        x_ulb_w = batch["unlabeled"]["img_ulb_w"] if "unlabeled" in batch else torch.empty((0, *x_lb.shape[1:]),
                                                                                           dtype=x_lb.dtype,
                                                                                           device=x_lb.device)
        x_ulb_s = batch["unlabeled"]["img_ulb_s"] if "unlabeled" in batch else torch.empty((0, *x_lb.shape[1:]),
                                                                                           dtype=x_lb.dtype,
                                                                                           device=x_lb.device)
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses

        x = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        outs = self(x)["multilabel"]

        logits_x_lb = outs[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = outs[num_lb:].chunk(2)

        sup_loss = self.heads["multilabel"].loss(logits_x_lb, y_lb)

        # # multiclass
        # probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

        # max_probs, _ = torch.max(probs_x_ulb_w.detach(), dim=-1)
        # mask = max_probs.ge(self.p_cutoff).to(max_probs.dtype)
        # pseudo_label = torch.argmax(probs_x_ulb_w, dim=-1)
        # unsup_loss = F.cross_entropy(logits_x_ulb_s, pseudo_label, reduction="none")

        # multilabel
        probs_x_ulb_w = torch.sigmoid(logits_x_ulb_w.detach())

        mask = probs_x_ulb_w.ge(self.p_cutoff).to(probs_x_ulb_w.dtype) + probs_x_ulb_w.le(1. - self.p_cutoff).to(probs_x_ulb_w.dtype)
        pseudo_label = probs_x_ulb_w.ge(0.5).to(probs_x_ulb_w.dtype)
        unsup_loss = F.binary_cross_entropy_with_logits(logits_x_ulb_s, pseudo_label, reduction="none")

        unsup_loss = unsup_loss * mask
        unsup_loss = unsup_loss.mean()
        total_loss = sup_loss + self.lambda_u * unsup_loss if not unsup_loss.isnan() else sup_loss

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": total_loss}


class DeFixMatchModel(FixMatchModel):
    STEP_OUTPUT = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x_lb = batch["labeled"]["img"]
        x_lb_s = batch["labeled"]["img_s"]
        y_lb = batch["labeled"]["multilabel"]

        x_ulb_w = batch["unlabeled"]["img_ulb_w"] if "unlabeled" in batch else torch.empty((0, *x_lb.shape[1:]),
                                                                                           dtype=x_lb.dtype,
                                                                                           device=x_lb.device)
        x_ulb_s = batch["unlabeled"]["img_ulb_s"] if "unlabeled" in batch else torch.empty((0, *x_lb.shape[1:]),
                                                                                           dtype=x_lb.dtype,
                                                                                           device=x_lb.device)
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        x = torch.cat((x_lb, x_lb_s, x_ulb_w, x_ulb_s))
        outs = self(x)["multilabel"]

        logits_x_lb, logits_x_lb_s = outs[:2 * num_lb].chunk(2)
        logits_x_ulb_w, logits_x_ulb_s = outs[2 * num_lb:].chunk(2)

        sup_loss = self.heads["multilabel"].loss(logits_x_lb, y_lb) + self.heads["multilabel"].loss(logits_x_lb_s, y_lb)
        sup_loss *= 0.5

        # multilabel
        probs_x_ulb_w = torch.sigmoid(logits_x_ulb_w.detach())

        mask = probs_x_ulb_w.ge(self.p_cutoff).to(probs_x_ulb_w.dtype) + probs_x_ulb_w.le(1. - self.p_cutoff).to(probs_x_ulb_w.dtype)
        pseudo_label = probs_x_ulb_w.ge(0.5).to(probs_x_ulb_w.dtype)

        unsup_loss = F.binary_cross_entropy_with_logits(logits_x_ulb_s, pseudo_label, reduction="none")
        unsup_loss = unsup_loss * mask
        unsup_loss = unsup_loss.mean()

        # debiasing
        probs_x_lb = torch.sigmoid(logits_x_lb.detach())
        mask_lb = probs_x_lb.ge(self.p_cutoff).to(probs_x_lb.dtype) + probs_x_lb.le(1. - self.p_cutoff).to(probs_x_lb.dtype)
        anti_pseudo_label = probs_x_lb.ge(0.5).to(probs_x_lb.dtype)

        anti_unsup_loss = F.binary_cross_entropy_with_logits(logits_x_lb, anti_pseudo_label, reduction="none")
        anti_unsup_loss = anti_unsup_loss * mask_lb
        anti_unsup_loss = anti_unsup_loss.mean()

        if not unsup_loss.isnan() and not anti_unsup_loss.isnan():
            total_loss = sup_loss + self.lambda_u * (unsup_loss - anti_unsup_loss)
        else:
            total_loss = sup_loss

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": total_loss}


