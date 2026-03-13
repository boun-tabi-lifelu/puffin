from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Linear, LazyLinear

from torch_geometric.nn import (
    GCNConv, DenseGraphConv, GATConv, DenseGATConv,
    dense_mincut_pool
)
from torch_geometric.utils import to_dense_batch, to_dense_adj

from proteinworkshop.models.graph_encoders.esm_embeddings import EvolutionaryScaleModeling
from proteinworkshop.types import EncoderOutput


class GNNType(str, Enum):
    GCN = "GCN"
    GAT = "GAT"


class FusionMethod(str, Enum):
    SUM = "sum"
    CAT = "cat"


# -------------------------
# ESM Handler (kept as-is)
# -------------------------
class ESMHandler(nn.Module):
    """Handles ESM embeddings and their projections."""
    def __init__(
        self,
        esm_model_path: Optional[str] = None,
        proj_layer: bool = False,
        esm_embed_dim: int = 128,
        input_feat_dim: Optional[int] = None,
        fuse_lm_method: FusionMethod = FusionMethod.SUM,
        use_only_esm: bool = False,
    ):
        super().__init__()
        self.esm_model_path = esm_model_path
        self.proj_layer = proj_layer
        self.esm_embed_dim = esm_embed_dim
        self.input_feat_dim = input_feat_dim
        self.fuse_lm_method = fuse_lm_method
        self.use_only_esm = use_only_esm

        self.esm_model: Optional[EvolutionaryScaleModeling] = None
        self.lm_proj: Optional[nn.Module] = None
        self.feat_proj: Optional[nn.Module] = None

        if self.esm_model_path:
            self._setup_esm_model()
            self._setup_projection_layers()

    def _setup_esm_model(self) -> None:
        model_path = Path(self.esm_model_path)
        self.esm_model = EvolutionaryScaleModeling(
            model_path.parent,
            model=model_path.name,
            mlp_post_embed=False,
            finetune=False,
        )
        for param in self.esm_model.model.parameters():
            param.requires_grad = False

    def _setup_projection_layers(self) -> None:
        if self.proj_layer:
            self.lm_proj = LazyLinear(self.esm_embed_dim)

        if not self.use_only_esm:
            # IMPORTANT: you may want to revisit feat_proj sizing.
            # Keeping your original logic.
            if self.input_feat_dim:
                self.feat_proj = LazyLinear(self.input_feat_dim)
            elif self.fuse_lm_method == FusionMethod.SUM:
                self.feat_proj = LazyLinear(self.esm_embed_dim)

    def forward(self, batch_: Any, x: Tensor) -> Tuple[Tensor, Tensor]:
        if hasattr(batch_, "esm_embeddings"):
            x_lm = batch_.esm_embeddings
        else:
            if self.esm_model is None:
                raise RuntimeError("ESM model not initialized but embeddings not provided.")
            x_lm = self.esm_model.esm_embed(batch_)

        if self.proj_layer and self.lm_proj is not None:
            x_lm = self.lm_proj(x_lm)

        if self.use_only_esm:
            return x_lm, x_lm

        if self.feat_proj is not None:
            x = self.feat_proj(x)

        if self.fuse_lm_method == FusionMethod.SUM:
            if x.size() != x_lm.size():
                raise ValueError(
                    f"Shape mismatch for SUM fusion: x={x.size()} vs x_lm={x_lm.size()}."
                )
            return F.relu(x + x_lm), x_lm

        if self.fuse_lm_method == FusionMethod.CAT:
            return F.relu(torch.cat([x, x_lm], dim=-1)), x_lm

        return x, x_lm


# -------------------------
# Dense (segment-level) GNN block
# -------------------------
class DenseSegGNNBlock(nn.Module):
    """
    Segment-level message passing on dense adjacency matrices (B, K, K).
    Uses DenseGraphConv or DenseGATConv.
    """
    def __init__(self, conv_cls: type, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv = conv_cls(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        pre = x
        x = self.norm(x)
        x = self.conv(x, adj)
        x = self.dropout(F.relu(x))
        return pre + x


# -------------------------
# Optional: segment -> residue cross-attn as segment updater
# -------------------------
class SegmentToResidueCrossAttn(nn.Module):
    """
    Segments query residue embeddings to update segment states.
    This does NOT create a bypass if you only read out from segments.
    """
    def __init__(
        self,
        seg_dim: int,
        res_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seg_dim = seg_dim
        self.res_dim = res_dim
        self.num_heads = num_heads

        # Project residues into seg_dim so attention works in one space
        self.res_proj = nn.Linear(res_dim, seg_dim) if res_dim != seg_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=seg_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(seg_dim)
        self.ff = nn.Sequential(
            nn.Linear(seg_dim, seg_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(seg_dim, seg_dim),
        )

    def forward(self, x_seg: Tensor, x_res: Tensor, res_mask: Tensor) -> Tensor:
        """
        Args:
            x_seg: [B, K, D]
            x_res: [B, N, Dr]
            res_mask: [B, N] (True for valid residues)
        """
        x_res = self.res_proj(x_res)  # [B, N, D]
        key_padding_mask = ~res_mask  # True for PAD positions
        # segments attend to residues
        attn_out, _ = self.attn(query=x_seg, key=x_res, value=x_res,
                                key_padding_mask=key_padding_mask, need_weights=False)
        x_seg = self.norm(x_seg + attn_out)
        x_seg = self.norm(x_seg + self.ff(x_seg))
        return x_seg


# -------------------------
# Readout (segments only, masked)
# -------------------------
def masked_mean_max_pool(x: Tensor, mask: Tensor) -> Tensor:
    """
    x: [B, K, D]
    mask: [B, K] bool (True = valid segment)
    returns: [B, 2D] (mean || max)
    """
    mask_f = mask.float().unsqueeze(-1)  # [B, K, 1]
    denom = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
    mean = (x * mask_f).sum(dim=1) / denom

    # max with masking
    x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    mx = x_masked.max(dim=1).values
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))

    return torch.cat([mean, mx], dim=-1)


class Puffin(nn.Module):
    """
    HARD BOTTLENECK VERSION:
      residues -> (MinCut) segments -> segment GNN -> segment readout -> classifier input

    Optional:
      segments -> residues cross-attn updater (still no bypass)
    """
    def __init__(
        self,
        gnn_type: GNNType,
        hidden_dim: int = 512,
        num_clusters: int = 64,
        num_res_gnn_layers: int = 2,
        num_seg_gnn_layers: int = 2,
        esm_model_path: Optional[str] = None,
        proj_layer: bool = False,
        esm_embed_dim: int = 128,
        input_feat_dim: Optional[int] = None,
        fuse_lm_method: FusionMethod = FusionMethod.SUM,
        use_only_esm: bool = False,
        # OPTION 2 toggle
        use_seg_res_cross_attn: bool = False,
        seg_res_attn_heads: int = 4,
        seg_res_source: str = "gnn",  # {"gnn","esm","fused"}
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.num_res_gnn_layers = num_res_gnn_layers
        self.num_seg_gnn_layers = num_seg_gnn_layers
        self.use_only_esm = use_only_esm

        self.use_seg_res_cross_attn = use_seg_res_cross_attn
        self.seg_res_attn_heads = seg_res_attn_heads
        assert seg_res_source in {"gnn", "esm", "fused"}
        self.seg_res_source = seg_res_source

        # ESM handler (optional)
        self.esm_handler = ESMHandler(
            esm_model_path=esm_model_path,
            proj_layer=proj_layer,
            esm_embed_dim=esm_embed_dim,
            input_feat_dim=input_feat_dim,
            fuse_lm_method=fuse_lm_method,
            use_only_esm=use_only_esm,
        )
        self.esm_embed_dim = esm_embed_dim
        self.fuse_lm_method = fuse_lm_method

        # Residue conv types
        conv_sparse, conv_dense = self._get_conv_types()

        # Residue input dim
        # If use_only_esm=True, residue features are ESM dim (esm_embed_dim)
        # Otherwise, conv1 takes -1 (PyG infers) or set explicitly if you prefer.
        res_input_dim = self.esm_embed_dim if self.use_only_esm else -1

        # Residue GNN (sparse)
        self.res_conv_in = conv_sparse(res_input_dim, self.hidden_dim)
        self.res_blocks = nn.ModuleList([
            self._make_sparse_block(conv_sparse, self.hidden_dim) for _ in range(self.num_res_gnn_layers)
        ])

        # Assignment head (on dense residues)
        self.assign_lin = Linear(self.hidden_dim, self.num_clusters)

        # Segment GNN (dense)
        self.seg_blocks = nn.ModuleList([
            DenseSegGNNBlock(conv_dense, self.hidden_dim, dropout=0.2)
            for _ in range(self.num_seg_gnn_layers)
        ])

        # Optional seg->res cross-attn updater
        if self.use_seg_res_cross_attn:
            # residue source dim depends on what you feed
            if self.seg_res_source == "gnn":
                res_dim = self.hidden_dim
            elif self.seg_res_source == "esm":
                res_dim = self.esm_embed_dim
            else:  # fused
                # if fused is CAT, residues may be larger; if SUM, same.
                # We'll infer at runtime via a small proj below:
                res_dim = None

            self._seg_res_resdim = res_dim  # store; handle fused in forward
            self.seg_res_attn = SegmentToResidueCrossAttn(
                seg_dim=self.hidden_dim,
                res_dim=self.hidden_dim if res_dim is None else res_dim,
                num_heads=self.seg_res_attn_heads,
            )

        # Final projection (classifier will use this embedding)
        self.final_lin = Linear(self.hidden_dim * 2, self.hidden_dim)

    def _get_conv_types(self) -> Tuple[type, type]:
        if self.gnn_type == GNNType.GCN:
            return GCNConv, DenseGraphConv
        return GATConv, DenseGATConv

    def _make_sparse_block(self, conv_cls: type, hidden_dim: int) -> nn.Module:
        # simple residual sparse block
        return nn.ModuleDict({
            "bn": nn.BatchNorm1d(hidden_dim),
            "conv": conv_cls(hidden_dim, hidden_dim),
            "drop": nn.Dropout(0.3),
        })

    def _forward_sparse_block(self, block: nn.ModuleDict, x: Tensor, edge_index: Tensor) -> Tensor:
        pre = x
        x = block["bn"](x)
        x = block["conv"](x, edge_index)
        x = block["drop"](F.relu(x))
        return pre + x

    def forward(self, batch_: Any, return_clusters: bool = False) -> EncoderOutput:
        x, edge_index, batch = batch_.x, batch_.edge_index, batch_.batch

        # Optional ESM fusion at residue level (does NOT bypass, because we won't read out residues directly)
        x_lm = None
        if (hasattr(self.esm_handler, "esm_model") and self.esm_handler.esm_model is not None) or hasattr(batch_, "esm_embeddings"):
            x, x_lm = self.esm_handler(batch_, x)  # x becomes fused residue features; x_lm is LM residue features

        # Residue GNN (sparse)
        x = F.relu(self.res_conv_in(x, edge_index))
        for blk in self.res_blocks:
            x = self._forward_sparse_block(blk, x, edge_index)

        # Make residues dense + adjacency dense (needed for MinCut)
        x_res_dense, res_mask = to_dense_batch(x, batch)          # [B, N, H], [B, N]
        adj_res = to_dense_adj(edge_index, batch)                 # [B, N, N]

        # Assignment logits and MinCut pooling
        s_logits = self.assign_lin(x_res_dense)                   # [B, N, K]
        # x_seg, adj_seg, m_loss, o_loss = dense_mincut_pool(x_res_dense, adj_res, s_logits, res_mask)

        # --- temperature-controlled assignment ---
        # temperature tau > 0, larger => softer assignments
        tau = getattr(self, "assign_temperature", 1.0)  # default if not set

        s_scaled = s_logits / max(tau, 1e-8)    # scale logits => softmax(logits/tau)
        x_seg, adj_seg, m_loss, o_loss = dense_mincut_pool(x_res_dense, adj_res, s_scaled, res_mask)

        # when you compute entropy/usage, use the same scaled logits
        s = F.softmax(s_scaled, dim=-1)         # [B, N, K]
        # x_seg: [B, K, H], adj_seg: [B, K, K]
        
         # [B, N, K]
        seg_mass = (s * res_mask.unsqueeze(-1).float()).sum(dim=1)  # [B, K]
        seg_mask = seg_mass > 1e-6                                # [B, K]

        # Optional: entropy diagnostics (not a training loss unless you weight it elsewhere)
        per_node_entropy = -(s * torch.log(s + 1e-9)).sum(dim=-1)  # [B, N]
        entropy_sharpness = (per_node_entropy * res_mask.float()).sum() / res_mask.float().sum().clamp(min=1.0)

        usage = seg_mass / seg_mass.sum(dim=-1, keepdim=True).clamp(min=1e-9)  # [B, K]
        usage_entropy = -(usage * torch.log(usage + 1e-9)).sum(dim=-1).mean()  # scalar

        diag = id_invariant_segmentation_diagnostics(
            s_logits=s_logits,
            res_mask=res_mask,
            tau=tau,
            active_mass_eps=1e-6,
            topk=5,
        )

        #  Update segments by attending to residues (via cross-attn) 
        if self.use_seg_res_cross_attn:
            if self.seg_res_source == "gnn":
                x_src = x_res_dense
                src_mask = res_mask
                # res_dim already hidden_dim
                # Use cross-attn normally (no detach)
                x_seg = self.seg_res_attn(x_seg, x_src, src_mask)
                # Alternatively, detach pooled segment embeddings to preserve bottleneck effect
                # x_seg_q = x_seg.detach()
                # x_seg = x_seg_q + self.seg_res_attn(x_seg_q, x_src, src_mask)
                # Detach query but not keys/values
                # x_seg = self.seg_res_attn(x_seg.detach(), x_src, src_mask)

            elif self.seg_res_source == "esm":
                if x_lm is None:
                    raise RuntimeError("seg_res_source='esm' requires ESM embeddings.")
                x_lm_dense, lm_mask = to_dense_batch(x_lm, batch)
                # Use cross-attn normally (no detach)
                x_seg = self.seg_res_attn(x_seg, x_lm_dense, lm_mask)
                # Detach pooled segment embeddings to preserve bottleneck effect
                # x_seg_q = x_seg.detach()
                # x_seg = x_seg_q + self.seg_res_attn(x_seg_q, x_lm_dense, lm_mask)
                # Detach query but not keys/values
                # x_seg = self.seg_res_attn(x_seg.detach(), x_lm_dense, lm_mask)

            else:  # fused
                # Build a fused residue tensor for attention keys/values
                if x_lm is None:
                    raise RuntimeError("seg_res_source='fused' requires ESM embeddings.")
                x_lm_dense, lm_mask = to_dense_batch(x_lm, batch)

                # fused choice: CAT is more expressive; SUM requires same dims.
                # We'll do CAT then project inside attention module via res_proj.
                x_fused = torch.cat([x_res_dense, x_lm_dense], dim=-1)  # [B, N, H+E]
                # If seg_res_attn was initialized with res_dim=hidden_dim (placeholder),
                # we need a matching module. Easiest: lazily create it once.
                if getattr(self, "_fused_attn_ready", False) is False:
                    self.seg_res_attn = SegmentToResidueCrossAttn(
                        seg_dim=self.hidden_dim,
                        res_dim=x_fused.size(-1),
                        num_heads=self.seg_res_attn_heads,
                    )
                    self._fused_attn_ready = True
                # Use cross-attn normally (no detach)
                x_seg = self.seg_res_attn(x_seg, x_fused, lm_mask & res_mask)
                # Detach pooled segment embeddings to preserve bottleneck effect
                # x_seg_q = x_seg.detach()
                # x_seg = x_seg_q + self.seg_res_attn(x_seg_q, x_fused, lm_mask & res_mask)
                # Detach query but not keys/values
                # x_seg = self.seg_res_attn(x_seg.detach(), x_fused, lm_mask & res_mask)

        # Segment-level GNN refinement (dense)
        for blk in self.seg_blocks:
            x_seg = blk(x_seg, adj_seg)

        # Segment-only readout (NO BYPASS)
        x_graph = masked_mean_max_pool(x_seg, seg_mask)           # [B, 2H]
        x_graph = F.relu(self.final_lin(x_graph))                 # [B, H]

        outputs: Dict[str, Any] = {
            "node_embedding": x_seg,              # note: now segment nodes, not residues
            "graph_embedding": x_graph,
            "mc_losses": [(m_loss, o_loss)],
            "entropy_loss": (entropy_sharpness.item(), usage_entropy.item()),
        }
        
        outputs["seg_diag"] = diag["batch"]      # dict of scalars
        outputs["seg_flags"] = diag["flags"]     # collapse rates

        if return_clusters:
            # Return hard assignments on residues (optional analysis; NOT used for bypass)
            hard = s.argmax(dim=-1)                                # [B, N]
            hard = hard.masked_fill(~res_mask, -1)
            outputs.update({"clusters": [hard], "seg_mask": [seg_mask]})

        return EncoderOutput(outputs)


import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

@torch.no_grad()
def id_invariant_segmentation_diagnostics(
    s_logits: torch.Tensor,          # [B, N, K]
    res_mask: torch.Tensor,          # [B, N] bool
    tau: float = 1.0,
    active_mass_eps: float = 1e-6,   # threshold to count a segment as "active"
    topk: int = 5,
) -> Dict[str, Any]:
    """
    ID-invariant segmentation diagnostics.

    Returns:
      - batch scalars (good for logging)
      - per-protein vectors (useful for histograms/debug)
    """
    assert s_logits.dim() == 3, f"s_logits must be [B,N,K], got {s_logits.shape}"
    assert res_mask.dim() == 2, f"res_mask must be [B,N], got {res_mask.shape}"
    B, N, K = s_logits.shape
    device = s_logits.device

    # Temperature-scaled soft assignments
    s_scaled = s_logits / max(float(tau), 1e-8)
    s = F.softmax(s_scaled, dim=-1)  # [B, N, K]

    # Mask to float
    m = res_mask.float()  # [B, N]
    denom_nodes = m.sum(dim=1).clamp(min=1.0)  # [B]

    # --- Per-node entropy (diffuse vs sharp) ---
    per_node_entropy = -(s * torch.log(s + 1e-9)).sum(dim=-1)     # [B, N]
    mean_node_entropy = (per_node_entropy * m).sum(dim=1) / denom_nodes  # [B]

    # --- Top-1 confidence and top1-top2 gap (sharpness proxy) ---
    top2 = torch.topk(s, k=2, dim=-1).values                      # [B, N, 2]
    top1 = top2[..., 0]                                           # [B, N]
    gap12 = top2[..., 0] - top2[..., 1]                           # [B, N]
    mean_top1 = (top1 * m).sum(dim=1) / denom_nodes               # [B]
    mean_gap12 = (gap12 * m).sum(dim=1) / denom_nodes             # [B]

    # --- Segment mass per protein (usage; ID-invariant as a multiset) ---
    # mass[b,k] = total probability assigned to segment k across residues in protein b
    seg_mass = (s * m.unsqueeze(-1)).sum(dim=1)                   # [B, K]
    total_mass = seg_mass.sum(dim=1).clamp(min=1e-9)              # [B] ~ number of residues
    seg_usage = seg_mass / total_mass.unsqueeze(-1)               # [B, K] sums to 1

    # active segments count
    active_segments = (seg_mass > active_mass_eps).sum(dim=1).float()  # [B]
    frac_active = active_segments / float(K)                           # [B]

    # largest segment fraction (collapse-to-1 detector)
    largest_seg_frac = seg_usage.max(dim=1).values                     # [B]

    # Gini-like concentration (optional; 0=uniform usage, high=concentrated)
    # (simple measure: L2 norm of usage)
    usage_l2 = torch.sqrt((seg_usage ** 2).sum(dim=1))                 # [B]
    # for uniform over K, usage_l2 ~ 1/sqrt(K); for single cluster, usage_l2 ~ 1

    # usage entropy (diversity across segments within each protein)
    usage_entropy = -(seg_usage * torch.log(seg_usage + 1e-9)).sum(dim=1)  # [B]
    # normalize to [0,1] by log(K) if you want comparability across K
    usage_entropy_norm = usage_entropy / torch.log(torch.tensor(float(K), device=device))

    # --- Segment size distribution stats (ID-invariant) ---
    # effective number of segments (perplexity): exp(H(usage))
    effective_segments = torch.exp(usage_entropy)                      # [B]
    # top-k usage fractions
    topk = min(int(topk), K)
    topk_usage = torch.topk(seg_usage, k=topk, dim=1).values          # [B, topk]
    top1_usage = topk_usage[:, 0]
    top3_usage_sum = topk_usage[:, :min(3, topk)].sum(dim=1)
    min_mass = seg_mass.min(dim=1).values.mean().item() # average min segment mass
    p10_mass = torch.quantile(seg_mass, 0.10, dim=1).mean().item() # avg 10th percentile mass
    # --- Batch-level summary (means) ---
    out: Dict[str, Any] = {}

    # Per-protein vectors (keep for debugging / optional histogram logging)
    out["per_protein"] = {
        "mean_node_entropy": mean_node_entropy,       # [B]
        "mean_top1": mean_top1,                       # [B]
        "mean_gap12": mean_gap12,                     # [B]
        "active_segments": active_segments,           # [B]
        "frac_active": frac_active,                   # [B]
        "largest_seg_frac": largest_seg_frac,         # [B]
        "usage_l2": usage_l2,                         # [B]
        "usage_entropy_norm": usage_entropy_norm,     # [B]
        "effective_segments": effective_segments,     # [B]
        "top1_usage": top1_usage,                     # [B]
        "top3_usage_sum": top3_usage_sum,             # [B]

    }
    # print(out["per_protein"])

    # Batch scalars (good for lightning self.log)
    out["batch"] = {
        "seg/mean_node_entropy": mean_node_entropy.mean().item(),
        "seg/mean_top1": mean_top1.mean().item(),
        "seg/mean_gap12": mean_gap12.mean().item(),
        "seg/active_segments": active_segments.mean().item(),
        "seg/frac_active": frac_active.mean().item(),
        "seg/largest_seg_frac": largest_seg_frac.mean().item(),
        "seg/usage_l2": usage_l2.mean().item(),
        "seg/usage_entropy_norm": usage_entropy_norm.mean().item(),
        "seg/effective_segments": effective_segments.mean().item(),
        "seg/top1_usage": top1_usage.mean().item(),
        "seg/top3_usage_sum": top3_usage_sum.mean().item(),
        "seg/min_segment_mass": min_mass,
        "seg/p10_segment_mass": p10_mass,
    }
    # print(out["batch"])
    # Convenience: "collapse flags" (ID-invariant heuristics)
    # Tune these thresholds with a few runs.
    collapse_hard = (largest_seg_frac > 0.90) | (active_segments <= 2)
    collapse_soft = (mean_node_entropy > 0.85 * torch.log(torch.tensor(float(K), device=device))) | (mean_gap12 < 0.05)

    out["flags"] = {
        "collapse_hard_rate": collapse_hard.float().mean().item(),
        "collapse_soft_rate": collapse_soft.float().mean().item(),
    }
    score, parts = seg_health_score(out["per_protein"], K)
    # print(score, parts)
    return out

import math
import torch

def seg_health_score(
    metrics: dict,
    K: int,
    # targets (good mid-range, not uniform, not collapsed)
    mu_H: float = 0.55,   sigma_H: float = 0.18,   # target normalized node entropy
    mu_u: float = 0.75,   sigma_u: float = 0.18,   # target normalized usage entropy
    mu_N: float = 0.25,   sigma_N: float = 0.12,   # target Neff/K  (0.25 => 16 of 64)
    # confidence thresholds
    tau_p: float = None, a_p: float = 40.0,        # top1 threshold, slope
    tau_d: float = 0.05, a_d: float = 60.0,        # gap12 threshold, slope
    # collapse threshold
    tau_c: float = 0.55, a_c: float = 25.0,
    # weights
    w_H: float = 1.0, w_p: float = 1.0, w_d: float = 1.0, w_u: float = 1.0, w_N: float = 1.0,
):
    """
    metrics: dict of per-protein tensors, each shape [B]
        required keys:
          mean_node_entropy, mean_top1, mean_gap12, usage_entropy_norm,
          effective_segments, largest_seg_frac
    returns:
        score_per_protein: [B] in (0,1)
        parts: dict of components (useful for debugging)
    """
    # pull tensors
    H   = metrics["mean_node_entropy"]
    p1  = metrics["mean_top1"]
    gap = metrics["mean_gap12"]
    Hu  = metrics["usage_entropy_norm"]
    Ne  = metrics["effective_segments"]
    fmx = metrics["largest_seg_frac"]

    device = H.device
    logK = math.log(K)

    # normalize node entropy
    Hn = H / logK  # [B], roughly in [0,1]

    # set a sensible default for top1 threshold if not given:
    # halfway between uniform (1/K) and a mild confident assignment (0.10)
    if tau_p is None:
        tau_p = float(0.5 * (1.0 / K + 0.10))

    # gaussian "good range" scores
    S_H = torch.exp(-0.5 * ((Hn - mu_H) / sigma_H) ** 2)
    S_u = torch.exp(-0.5 * ((Hu - mu_u) / sigma_u) ** 2)
    S_N = torch.exp(-0.5 * (((Ne / K) - mu_N) / sigma_N) ** 2)

    # sigmoid confidence
    S_p = torch.sigmoid(a_p * (p1 - tau_p))
    S_d = torch.sigmoid(a_d * (gap - tau_d))

    # collapse penalty
    P_col = torch.sigmoid(a_c * (fmx - tau_c))  # near 0 if safe, near 1 if collapsed
    safe = 1.0 - P_col

    # weighted geometric mean (stable)
    ws = torch.tensor([w_H, w_p, w_d, w_u, w_N], device=device)
    comps = torch.stack([S_H, S_p, S_d, S_u, S_N], dim=0)  # [5, B]
    eps = 1e-8
    log_gm = (ws[:, None] * torch.log(comps.clamp_min(eps))).sum(dim=0) / ws.sum()
    gm = torch.exp(log_gm)

    score = (gm * safe).clamp(0.0, 1.0)

    parts = {
        "S_H": S_H, "S_p": S_p, "S_d": S_d, "S_u": S_u, "S_N": S_N,
        "P_col": P_col, "safe": safe,
        "Hn": Hn,
    }
    return score, parts
