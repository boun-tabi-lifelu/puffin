import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torch import Tensor
from typing import Any
from jaxtyping import Float

class AUPRC(Metric):
    """Class for AUPRC metric.

    :param compute_on_cpu: Whether to compute the metric on CPU,
            defaults to ``True``.
    :type compute_on_cpu: bool, optional
    """

    def __init__(self, num_classes: int, compute_on_cpu: bool = True) -> None:
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.num_classes = num_classes
        self.compute_on_cpu = compute_on_cpu

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the current state of the metric.

        :param preds: Tensor of predictions
        :type preds: torch.Tensor
        :param target: Tensor of targets
        :type target: torch.Tensor
        """
        self.preds.append(preds.detach())
        self.targets.append(target.detach())

    def compute(self) -> torch.Tensor:
        """Computes the AUPRC metric.

        :return: AUPRC metric value
        :rtype: torch.Tensor
        """
        return self.macro_auprc(
            torch.sigmoid(torch.cat(self.preds)), torch.cat(self.targets), self.num_classes
        )

    def auprc(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Area under precision-recall curve.

        Parameters:
        :param pred: predictions. Shape ``(N,)``
        :param target: binary targets of shape ``(N,)``
        """
        eps = 1e-10
        order = pred.argsort(descending=True)
        target = target[order]
        precision = target.cumsum(0) / torch.arange(
            1, len(target) + 1, device=target.device
        )
        return precision[target == 1].sum() / ((target == 1).sum() + eps)

    def macro_auprc(self, pred: torch.Tensor, target: torch.Tensor, num_classes:int) -> torch.Tensor:
        """
        Computes macro-averaged AUPRC for multi-class classification.

        :param pred: predictions of shape (N, num_classes), containing confidence scores for each class
        :param target: binary targets of shape (N, num_classes) or integer targets of shape (N,)
        :param num_classes: number of classes
        :return: macro-averaged AUPRC
        """
        auprc_per_class = []
        for c in range(num_classes):
            pred_c = pred[:, c]
            target_c = (target == c).float() if target.ndim == 1 else target[:, c]
            
            auprc_c = self.auprc(pred_c, target_c)  
            auprc_per_class.append(auprc_c)

        return torch.stack(auprc_per_class).mean()



class F1Max(Metric):
    """
    Implements the protein-centric F1 Max metric.
    """

    def __init__(self, num_classes: int, compute_on_cpu: bool = True) -> None:
        """Initializes the F1Max metric.

        :param num_classes: Number of classes.
        :type num_classes: int
        :param compute_on_cpu: Whether to compute the metric on CPU,
            defaults to ``True``.
        :type compute_on_cpu: bool, optional
        """
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.compute_on_cpu = compute_on_cpu

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds.detach())
        self.targets.append(target.detach())

    def compute(self) -> Any:
        """Computes the F1Max metric.

        .. seealso::
            :py:class:`proteinworkshop.metrics.f1_max.F1Max.f1_max`

        :return: F1Max metric value
        :rtype: Any
        """
        if self.preds[0].ndim == 2:
            # return self.f1_max(torch.stack(self.preds), torch.stack(self.targets))
            return self.f1_max(torch.cat(self.preds), torch.cat(self.targets))
        return self.f1_max(
            torch.cat(self.preds, dim=0), torch.cat(self.targets, dim=0)
        )

    def f1_max(
        self,
        pred: Float[Tensor, "batch classes"],
        target: Float[Tensor, "batch classes"],
    ):
        """
        F1 score with the optimal threshold.
        This function first enumerates all possible thresholds for deciding
        positive and negative samples, and then pick the threshold with the
        maximal F1 score.

        Parameters:
            pred (Tensor): predictions of shape :math:`(B, N)`
            target (Tensor): binary targets of shape :math:`(B, N)`
        """
        pred = torch.sigmoid(pred)

        if target.ndim == 1:
            target = F.one_hot(
                target.long(), num_classes=pred.shape[1]
            ).float()
        order = pred.argsort(descending=True, dim=1)
        target = target.gather(1, order).int()
        precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
        recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
        is_start = torch.zeros_like(target).bool()
        is_start[:, 0] = 1
        is_start = torch.scatter(is_start, 1, order, is_start)

        all_order = pred.flatten().argsort(descending=True)
        order = (
            order
            + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
            * order.shape[1]
        )
        order = order.flatten()
        inv_order = torch.zeros_like(order)
        inv_order[order] = torch.arange(order.shape[0], device=order.device)
        is_start = is_start.flatten()[all_order]
        all_order = inv_order[all_order]
        precision = precision.flatten()
        recall = recall.flatten()
        all_precision = precision[all_order] - torch.where(
            is_start, torch.zeros_like(precision), precision[all_order - 1]
        )
        all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
        all_recall = recall[all_order] - torch.where(
            is_start, torch.zeros_like(recall), recall[all_order - 1]
        )
        all_recall = all_recall.cumsum(0) / pred.shape[0]
        all_f1 = (
            2
            * all_precision
            * all_recall
            / (all_precision + all_recall + 1e-10)
        )
        return all_f1.max()