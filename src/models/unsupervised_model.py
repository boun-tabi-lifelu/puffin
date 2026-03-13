import torch
import torch.distributed as torch_dist

import graphein
import lovely_tensors as lt
from typing import Literal, Optional, Union
from loguru import logger as log
from omegaconf import DictConfig
from proteinworkshop.models.base import BenchMarkModel
from proteinworkshop.types import EncoderOutput, ModelOutput
from torch_geometric.data import Batch
from graphein.protein.tensor.data import ProteinBatch

graphein.verbose(False)
lt.monkey_patch()

class UnsupervisedModel(BenchMarkModel):
    """Thin wrapper around BenchMarkModel exposing unsupervised clustering losses."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def forward(
        self, batch: Union[Batch, ProteinBatch], perturbed: bool = False
    ) -> ModelOutput:
        """Return the Monte Carlo loss tuples produced by the encoder."""
        output: EncoderOutput = self.encoder(batch, perturbed)
        return output["mc_losses"]

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

    def _do_step(
        self,
        batch: Union[Batch, ProteinBatch],
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        """Compute and log the MC losses for a single step and return the total."""
        mc_losses = self(batch)
        loss_prefix = f"{stage}/loss"
        total_key = f"{loss_prefix}/total"
        loss = {total_key: torch.tensor(0.0, device=self.device)}

        for idx, (m_loss, o_loss) in enumerate(mc_losses):
            loss[f"{loss_prefix}/m_loss{idx}"] = m_loss
            loss[f"{loss_prefix}/o_loss{idx}"] = o_loss
            loss[f"{loss_prefix}/mc_loss{idx}"] = m_loss + o_loss
            loss[total_key] = loss[total_key] + m_loss + o_loss

        self.log_dict(loss, prog_bar=True)
        return loss[total_key]

    def _do_step_catch_oom(
        self,
        batch: Union[Batch, ProteinBatch],
        batch_idx: int,
        stage: Literal["train", "val"],
    ) -> Optional[torch.Tensor]:
        """Same as `_do_step` but skips batches if an OOM occurs during the forward pass."""
        skip_flag = torch.zeros(
            (), device=self.device, dtype=torch.bool
        )  # NOTE: for skipping batches in a multi-device setting

        try:
            mc_losses = self(batch)
            loss_prefix = f"{stage}/loss"
            total_key = f"{loss_prefix}/total"
            loss = {total_key: torch.tensor(0.0, device=self.device)}

            for idx, (m_loss, o_loss) in enumerate(mc_losses):
                loss[f"{loss_prefix}/m_loss{idx}"] = m_loss
                loss[f"{loss_prefix}/o_loss{idx}"] = o_loss
                loss[f"{loss_prefix}/mc_loss{idx}"] = m_loss + o_loss
                loss[total_key] = loss[total_key] + m_loss + o_loss

            self.log_dict(loss, prog_bar=True)
            return loss[total_key]

        except Exception as e:
            skip_flag = torch.ones((), device=self.device, dtype=torch.bool)

            if "out of memory" in str(e):
                log.warning(
                    f"Ran out of memory in the forward pass. Skipping current {stage} batch with index {batch_idx}."
                )
                if not torch_dist.is_initialized():
                    if self.training:
                        for p in self.trainer.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                    return None
            else:
                if not torch_dist.is_initialized():
                    raise e

        if torch_dist.is_initialized():
            # CREDIT: Lightning issue discussion for skip synchronization.
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
                log.warning(
                    f"Failed to perform the forward pass for at least one rank. Skipping {stage} batches for all ranks."
                )
                return None

        return None
