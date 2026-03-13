
import collections
import os
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch_geometric.transforms as T
import torchmetrics
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple
from lightning import LightningModule
from omegaconf import DictConfig
from proteinworkshop.datasets.utils import create_example_batch
from torch_scatter import scatter

import copy
from typing import List

import graphein
import hydra
import lightning as L
import lovely_tensors as lt
import torch
import torch.nn as nn
import torch_geometric
from torch.nn import functional as F

from loguru import logger as log

from proteinworkshop import (
    utils,
)
from proteinworkshop.models.base import BenchMarkModel
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from proteinworkshop.types import ModelOutput, EncoderOutput
from omegaconf import DictConfig
from typing import Literal, Union
graphein.verbose(False)
lt.monkey_patch()


import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, segment_embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segment_embeddings: [B, K, D] - segment-level representations
            labels: [B, C] - binary multi-labels (GO terms)
        Returns:
            InfoNCE loss encouraging function-specific segment clusters
        """
        B, K, D = segment_embeddings.shape
        device = segment_embeddings.device

        # Flatten segments
        segments = segment_embeddings.view(B * K, D)  # [B*K, D]
        segments = F.normalize(segments, dim=-1)  # unit vectors

        # Repeat labels for each segment
        labels = labels.unsqueeze(1).repeat(1, K, 1).view(B * K, -1)  # [B*K, C]

        # Compute cosine similarity matrix: [B*K x B*K]
        sim_matrix = torch.matmul(segments, segments.T) / self.temperature

        # Determine which segment pairs are positive (label overlap)
        with torch.no_grad():
            label_sim = torch.matmul(labels.float(), labels.T.float())  # [B*K, B*K]
            positives = label_sim > 0  # binary mask

            # Remove self-comparisons
            diag = torch.eye(B * K, dtype=torch.bool, device=device)
            positives[diag] = False

        # Binary cross-entropy between similarity and functional similarity
        loss = F.binary_cross_entropy_with_logits(sim_matrix, positives.float())

        return loss


class DualModel(BenchMarkModel):
    """
    DualModel integrates multiple objectives for protein function prediction
    and segment discovery. It combines:

    - Function prediction loss (e.g., multilabel GO classification)
    - Unit/cluster prediction loss (e.g., MinCut pooling)
    - Mutual information loss via InfoNCE between segments and GO terms
    - Entropy-based regularization to encourage sharp yet diverse cluster usage
    """

    def __init__(self,
                 cfg: DictConfig,
                 function_weight: float = 0.8,
                 unit_weight: float = 0.1,
                 mutual_weight: float = 0.1,
                 mutual_temp: float = 0.7,
                 entropy_weight: float = 0.0,
                 entropy_alpha: float = 1.0,
                 entropy_beta: float = 0.3) -> None:
        """
        Args:
            cfg (DictConfig): Hydra configuration object.
            function_weight (float): Weight of function classification loss.
            unit_weight (float): Weight of unit (segment) loss like MinCut.
            mutual_weight (float): Weight of InfoNCE mutual information loss.
            mutual_temp (float): Temperature for InfoNCE loss.
            entropy_weight (float): Weight of entropy regularization loss.
            entropy_alpha (float): Sharpness term coefficient (minimize entropy per node).
            entropy_beta (float): Diversity term coefficient (maximize entropy across clusters).
        """
        super().__init__(cfg)

        self.function_weight = float(function_weight)
        self.unit_weight = float(unit_weight)
        self.mutual_weight = float(mutual_weight)
        self.entropy_weight = float(entropy_weight)

        self.entropy_alpha = float(entropy_alpha)
        self.entropy_beta = float(entropy_beta)

        self.segment_info_nce = SegmentInfoNCELoss(temperature=mutual_temp)

        # ---- Schedules ----
        self.unit_weight_min = float(cfg.get("schedule.unit_weight_min", 0.0))
        self.unit_weight_max = float(cfg.get("schedule.unit_weight_max", self.unit_weight))

        self.warmup_epochs = int(cfg.get("schedule.warmup_epochs", 5))
        self.ramp_epochs = int(cfg.get("schedule.ramp_epochs", 10))

        self.temp_start = float(cfg.get("schedule.temp_start", 2.0))   # softer
        self.temp_end = float(cfg.get("schedule.temp_end", 1.0))       # sharper
        self.temp_warmup_epochs = int(cfg.get("schedule.temp_warmup_epochs", self.warmup_epochs))
        self.temp_ramp_epochs = int(cfg.get("schedule.temp_ramp_epochs", self.ramp_epochs))

 
        log.info(
            f"Training with a mixture of objectives: "
            f"function_weight={self.function_weight}, "
            f"unit_weight={self.unit_weight}, "
            f"mutual_weight={self.mutual_weight}, "
            f"entropy_weight={self.entropy_weight}"
            f" (unit weight schedule: {self.unit_weight_min} -> {self.unit_weight_max} over "
            f"{self.warmup_epochs + self.ramp_epochs} epochs)"
        )



    def forward(self, batch: Union[Batch, ProteinBatch], perturbed=False) -> ModelOutput:
        # temperature schedule only matters during training; for val/test you can freeze at temp_end
        tau = self._current_temperature() if self.training else self.temp_end
        if hasattr(self.encoder, "assign_temperature"):
            self.encoder.assign_temperature = tau
        else:
            # safe fallback: dynamically attach
            setattr(self.encoder, "assign_temperature", tau)

        output: EncoderOutput = self.encoder(batch, perturbed)

        # assume you returned it in EncoderOutput:
        # seg_diag = output.get("seg_diag", None)
        # seg_flags = output.get("seg_flags", None)

        # if seg_diag is not None:
        #     for k, v in seg_diag.items():
        #         self.log(k, v, prog_bar=False, on_step=True, on_epoch=True, batch_size=batch.num_graphs)

        # if seg_flags is not None:
        #     for k, v in seg_flags.items():
        #         self.log(f"seg/{k}", v, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch.num_graphs)


        output = self.transform_encoder_output(output, batch)

        if self.decoder is not None:
            for output_head in self.config.decoder.keys():
                if hasattr(self.decoder[output_head], "requires_pos"):
                    output[output_head] = self.decoder[output_head](
                        edge_index=batch.edge_index,
                        scalar_features=output["node_embedding"],
                        pos=batch.pos,
                    )
                else:
                    emb_type = self.decoder[
                        output_head
                    ].input  # node_embedding or graph_embedding
                    output[output_head] = self.decoder[output_head](
                        output[emb_type]
                    )
        return self.compute_output(output, batch), output["graph_embedding"], output["node_embedding"], output["mc_losses"],  output["entropy_loss"]

    #def compute_loss(self, y_hat, y):
    #    loss = {k: v(y_hat[k], y[k]) for k, v in self.losses.items()}
    #
    #    # Scale loss terms by coefficient
    #    if self.config.get("task.aux_loss_coefficient"):
    #        for (output, coefficient) in self.config.task.aux_loss_coefficient.items():
    #            loss[output] = coefficient * loss[output]
    #
    #    loss["total"] = sum(loss.values())
    #    return loss

    def _do_step(self, batch, batch_idx, stage: Literal["train", "val", "test"]) -> torch.Tensor:

        # scheduled weights
        unit_w = self._current_unit_weight() if self.training else self.unit_weight_max
        tau = self._current_temperature() if self.training else self.temp_end
        # log schedules (once per epoch is also fine)
        self.log("schedule/unit_weight", unit_w, prog_bar=True, on_step=True, on_epoch=True)
        self.log("schedule/temperature", tau, prog_bar=True, on_step=True, on_epoch=True)

        # Get true labels
        y = self.get_labels(batch)

        # Forward pass: output includes graph prediction, graph embeddings, segment embeddings, mincut losses, entropy terms
        y_hat, g_feat, seg_feat, mc_losses, entropy_loss = self(batch)

        # Compute supervised function prediction loss
        loss = self.compute_loss(y_hat, y)
        total_loss = self.function_weight * loss["graph_label"]

        # Segment clustering (mincut) loss
        if unit_w != 0:
            total_mc_loss = 0
            for ix, (m_loss, o_loss) in enumerate(mc_losses):
                loss[f"m_loss{ix}"] = m_loss
                loss[f"o_loss{ix}"] = o_loss
                loss[f"mc_loss{ix}"] = m_loss + o_loss
                total_mc_loss += m_loss + o_loss
            loss["total_mc_loss"] = total_mc_loss
            total_loss += unit_w * total_mc_loss

        # Segment entropy regularization
        # if self.entropy_weight != 0:
        #     entropy_sharpness, entropy_diversity = entropy_loss
        #     entropy_reg = self.entropy_alpha * entropy_sharpness - self.entropy_beta * entropy_diversity
        #     loss["segment_entropy"] = entropy_reg
        #     total_loss += self.entropy_weight * entropy_reg

        # # InfoNCE mutual information loss
        # if self.mutual_weight != 0:
        #     mutual_loss = self.segment_info_nce(seg_feat, y["graph_label"].float())
        #     loss["segment_info_nce"] = mutual_loss
        #     total_loss += self.mutual_weight * mutual_loss

        # Final loss
        loss["total"] = total_loss

        # Logging
        self.log_metrics(loss, y_hat, y, stage, batch=batch)

        return total_loss

            
    def _do_step_catch_oom(self, batch, batch_idx, stage: Literal["train", "val"]) -> Optional[torch.Tensor]:
        # By default, do not skip the current batch
        skip_flag = torch.zeros((), device=self.device, dtype=torch.bool)

        unit_w = self._current_unit_weight() if self.training else self.unit_weight_max
        tau = self._current_temperature() if self.training else self.temp_end
        print(f"Current unit weight: {unit_w}, temperature: {tau}")


        # log schedules (once per epoch is also fine)
        self.log("schedule/unit_weight", unit_w, prog_bar=True, on_step=False, on_epoch=True)
        self.log("schedule/temperature", tau, prog_bar=True, on_step=False, on_epoch=True)
        try:
            y = self.get_labels(batch)
            y_hat, g_feat, seg_feat, mc_losses, entropy_loss = self(batch)
            loss = self.compute_loss(y_hat, y)

            total_loss = self.function_weight * loss["graph_label"]

            # Segment clustering (MinCut) loss
            if unit_w != 0:
                total_mc_loss = 0
                for ix, (m_loss, o_loss) in enumerate(mc_losses):
                    loss[f"m_loss{ix}"] = m_loss
                    loss[f"o_loss{ix}"] = o_loss
                    loss[f"mc_loss{ix}"] = m_loss + o_loss
                    total_mc_loss += m_loss + o_loss
                loss["total_mc_loss"] = total_mc_loss
                total_loss += unit_w * total_mc_loss

            # Entropy regularization loss
            # if self.entropy_weight != 0:
            #     entropy_sharpness, entropy_diversity = entropy_loss
            #     entropy_reg = self.entropy_alpha * entropy_sharpness - self.entropy_beta * entropy_diversity
            #     loss["segment_entropy"] = entropy_reg
            #     total_loss += self.entropy_weight * entropy_reg

            # # InfoNCE loss for function-specific segments
            # if self.mutual_weight != 0:
            #     mutual_loss = self.segment_info_nce(seg_feat, y["graph_label"].float())
            #     loss["segment_info_nce"] = mutual_loss
            #     total_loss += self.mutual_weight * mutual_loss

            loss["total"] = total_loss
            self.log_metrics(loss, y_hat, y, stage, batch=batch)
            return total_loss

        except Exception as e:
            skip_flag = torch.ones((), device=self.device, dtype=torch.bool)

            if "out of memory" in str(e).lower():
                log.warning(f"OOM error in {stage} batch {batch_idx}. Skipping batch.")
                if not torch_dist.is_initialized() and self.training:
                    for p in self.trainer.model.parameters():
                        if p.grad is not None:
                            del p.grad
                return None
            else:
                if not torch_dist.is_initialized():
                    raise e

        # Handle batch skipping in multi-GPU setting
        if torch_dist.is_initialized():
            world_size = torch_dist.get_world_size()
            torch_dist.barrier()
            result = [torch.zeros_like(skip_flag) for _ in range(world_size)]
            torch_dist.all_gather(result, skip_flag)
            any_skipped = torch.sum(torch.stack(result)).bool().item()
            if any_skipped:
                if self.training:
                    for p in self.trainer.model.parameters():
                        if p.grad is not None:
                            del p.grad
                log.warning(f"Batch skipped on at least one rank during {stage}.")
                return None

        return loss["total"]


    def training_step(
        self, batch: Union[Batch, ProteinBatch], batch_idx: int
    ) -> Optional[torch.Tensor]:
      
        return self._do_step_catch_oom(batch, batch_idx, "train")

    def validation_step(
        self, batch: Union[Batch, ProteinBatch], batch_idx: int
    ) -> Optional[torch.Tensor]:

        return self._do_step_catch_oom(batch, batch_idx, "val")

    def test_step(
        self, batch: Union[Batch, ProteinBatch], batch_idx: int
    ) -> torch.Tensor:

        return self._do_step(batch, batch_idx, "test")

    def _linear_ramp(self, epoch: int, warmup: int, ramp: int, start: float, end: float) -> float:
        """Piecewise linear: [0..warmup) => start, [warmup..warmup+ramp] => ramp to end, then end."""
        #print(f"Linear ramp: epoch={epoch}, warmup={warmup}, ramp={ramp}, start={start}, end={end}")
        if ramp <= 0:
            #print("No ramp phase, returning end value.", end)
            return end if epoch >= warmup else start
        if epoch < warmup:
            #print("In warmup phase, returning start value.", start)
            return start
        if epoch >= warmup + ramp:
            #print("After ramp phase, returning end value.", end)
            return end
        #print("In ramp phase, interpolating value.")
        t = (epoch - warmup) / float(ramp)
        return start + t * (end - start)

    def _current_unit_weight(self) -> float:
        # warmup: unit_weight_min; ramp -> unit_weight_max
        return self._linear_ramp(
            epoch=int(self.current_epoch),
            warmup=self.warmup_epochs,
            ramp=self.ramp_epochs,
            start=self.unit_weight_min,
            end=self.unit_weight_max,
        )


    def _current_temperature(self) -> float:
        # start high (soft), anneal to low (sharp)
        return self._linear_ramp(
            epoch=int(self.current_epoch),
            warmup=self.temp_warmup_epochs,
            ramp=self.temp_ramp_epochs,
            start=self.temp_start,
            end=self.temp_end,
        )
