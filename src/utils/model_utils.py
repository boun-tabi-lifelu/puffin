import collections
from typing import Optional, Union

import torch
from loguru import logger as log
from proteinworkshop.models.base import BenchMarkModel

from src.models.dual_model import DualModel
from src.models.unsupervised_model import UnsupervisedModel


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_state_dict(ckpt_path: str, device: torch.device):
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint)!r}.")
    return checkpoint.get("state_dict", checkpoint)


def _prefixed_weights(state_dict, prefix: str):
    weights = collections.OrderedDict()
    prefixes = (f"{prefix}.", f"model.{prefix}.")
    for key, value in state_dict.items():
        for full_prefix in prefixes:
            if key.startswith(full_prefix):
                weights[key.replace(full_prefix, "", 1)] = value
                break
    return weights


def load_model(cfg, batch, device: Optional[Union[str, torch.device]] = None):
    device = _resolve_device(device)
    log.info(f"Instantiating {cfg.objective_type} model on {device}")

    if cfg.objective_type == "unsupervised":
        model = UnsupervisedModel(cfg)
    elif cfg.objective_type == "dual":
        model = DualModel(
            cfg,
            cfg.get("function_weight", 0.9),
            cfg.get("unit_weight", 0.1),
        )
    else:
        model = BenchMarkModel(cfg)

    model = model.to(device)
    model.eval()

    log.info("Initializing lazy layers...")
    with torch.no_grad():
        batch = batch.to(device)
        log.info(f"Unfeaturized batch: {batch}")
        batch = model.featurise(batch)
        log.info(f"Featurized batch: {batch}")
        out = model.forward(batch)
        log.info(f"Model output: {out}")
        del batch, out

    log.info(f"Loading weights from checkpoint {cfg.ckpt_path}...")
    state_dict = _load_state_dict(cfg.ckpt_path, device)

    encoder_weights = _prefixed_weights(state_dict, "encoder")
    if encoder_weights:
        err = model.encoder.load_state_dict(encoder_weights, strict=False)
        log.info(f"Loading encoder weights: {err}")
    else:
        log.warning("No encoder weights found in checkpoint.")

    decoder_weights = _prefixed_weights(state_dict, "decoder")
    if decoder_weights and getattr(model, "decoder", None) is not None:
        err = model.decoder.load_state_dict(decoder_weights, strict=False)
        log.info(f"Loading decoder weights: {err}")
    elif decoder_weights:
        log.warning("Decoder weights found, but model has no decoder.")
    else:
        log.warning("No decoder weights found in checkpoint.")

    return model.to(device)
