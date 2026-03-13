import torch
import collections
from proteinworkshop.models.base import BenchMarkModel
from src.models.unsupervised_model import UnsupervisedModel
from src.models.contrastive_model import ContrastiveModel
from src.models.dual_model import DualModel
from loguru import logger as log

def load_model(cfg, batch):
    log.info(f"Instantiating {cfg.objective_type} model")
    if cfg.objective_type == "contrastive": 
        model = ContrastiveModel(cfg, cfg.tau, cfg.alpha)
    elif cfg.objective_type == "unsupervised":
        model = UnsupervisedModel(cfg)
    elif cfg.objective_type == "dual":
        model = DualModel(cfg, cfg.lambda_w)
    else: 
        model = BenchMarkModel(cfg)


    log.info("Initializing lazy layers...")
    with torch.no_grad():
        # batch = next(iter(dataloader))
        log.info(f"Unfeaturized batch: {batch}")
        batch = model.featurise(batch)
        log.info(f"Featurized batch: {batch}")
        out = model.forward(batch)
        log.info(f"Model output: {out}")
        del batch, out

    # Load weights
    # We only want to load weights
    log.info(f"Loading weights from checkpoint {cfg.ckpt_path}...")
    state_dict = torch.load(cfg.ckpt_path)["state_dict"]

    encoder_weights = collections.OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("encoder"):
            encoder_weights[k.replace("encoder.", "")] = v
    err = model.encoder.load_state_dict(encoder_weights, strict=False)
    log.info(f"Loading encoder weights: {err}")

    decoder_weights = collections.OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("decoder"):
            decoder_weights[k.replace("decoder.", "")] = v
    err = model.decoder.load_state_dict(decoder_weights, strict=False)
    log.info(f"Loading decoder weights: {err}")
    model = model.cuda()
    return model 