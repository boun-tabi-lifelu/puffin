# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import ssl
import copy
import hydra
import lightning as L
import rootutils
from beartype.typing import List, Optional
from lightning import Callback, LightningDataModule, Trainer
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies.strategy import Strategy
from omegaconf import DictConfig
from proteinworkshop import register_custom_omegaconf_resolvers
from proteinworkshop.models.base import BenchMarkModel

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import (
    register_custom_omegaconf_resolvers as src_register_custom_omegaconf_resolvers,
)
from src.models.contrastive_model import ContrastiveModel
from src.models.unsupervised_model import UnsupervisedModel
from src.models.dual_model import DualModel
from src.models.spectral_model import SpectralModel
from src.models.heal_contrastive_learning import HEALContrastiveModel
from src import resolve_omegaconf_variable
from src.utils import (
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    #log_hyperparameters,
    task_wrapper,
)
from loguru import logger as log
from graphein.protein.tensor.dataloader import ProteinDataLoader
def _num_training_steps(
    train_dataset: ProteinDataLoader, trainer: L.Trainer
) -> int:
    """
    Returns total training steps inferred from datamodule and devices.

    :param train_dataset: Training dataloader
    :type train_dataset: ProteinDataLoader
    :param trainer: Lightning trainer
    :type trainer: L.Trainer
    :return: Total number of training steps
    :rtype: int
    """
    if trainer.max_steps != -1:
        return trainer.max_steps

    dataset_size = (
        trainer.limit_train_batches
        if trainer.limit_train_batches not in {0, 1}
        else len(train_dataset) * train_dataset.batch_size
    )

    log.info(f"Dataset size: {dataset_size}")

    num_devices = max(1, trainer.num_devices)
    effective_batch_size = (
        train_dataset.batch_size
        * trainer.accumulate_grad_batches
        * num_devices
    )
    return (dataset_size // effective_batch_size) * trainer.max_epochs

def train(
    cfg: DictConfig
):  # sourcery skip: extract-method
    print(cfg)
    """
    Trains a model from a config.

    1. The datamodule is instantiated from ``cfg.dataset.datamodule``.
    2. The callbacks are instantiated from ``cfg.callbacks``.
    3. The logger is instantiated from ``cfg.logger``.
    4. The trainer is instantiated from ``cfg.trainer``.
    5. (Optional) If the config contains a scheduler, the number of training steps is
         inferred from the datamodule and devices and set in the scheduler.
    6. The model is instantiated from ``cfg.model``.
    7. The datamodule is setup and a dummy forward pass is run to initialise
    lazy layers for accurate parameter counts.
    8. Hyperparameters are logged to wandb if a logger is present.
    9. The model is compiled if ``cfg.compile`` is True.
    10. The model is trained if ``cfg.task_name`` is ``"train"``.
    11. The model is tested if ``cfg.test`` is ``True``.

    :param cfg: DictConfig containing the config for the experiment
    :type cfg: DictConfig
    :param encoder: Optional encoder to use instead of the one specified in
        the config
    :type encoder: Optional[nn.Module]
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.dataset.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset.datamodule)

    #log.info(f"Instantiating model <{cfg.model._target_}>")
    #with open_dict(cfg):
    #    cfg.model.model_cfg = validate_config(cfg.model.model_cfg)

    if cfg.objective_type == "contrastive": 
        log.info("Instantiating contrastive model")
        model = ContrastiveModel(cfg, cfg.tau, cfg.alpha)
    elif cfg.objective_type == "unsupervised":
        log.info("Instantiating unsupervised model")
        model = UnsupervisedModel(cfg)
    elif cfg.objective_type == 'dual':
        log.info("Instantiating dual model")
     
        model = DualModel(
            cfg, 
            cfg.function_weight, 
            cfg.unit_weight, 
            cfg.mutual_weight,
            cfg.mutual_temp,
            cfg.entropy_weight,
            cfg.entropy_alpha,
            cfg.entropy_beta
        )
    elif cfg.objective_type == 'spectral':
        log.info("Instantiating dual model")
        model = SpectralModel(cfg, cfg.lambda_w)
    else:
        log.info("Instantiating benchmark model")
        model = BenchMarkModel(cfg)
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    plugins = None
    if "_target_" in cfg.environment:
        log.info(f"Instantiating environment <{cfg.environment._target_}>")
        plugins: ClusterEnvironment = hydra.utils.instantiate(cfg.environment)

    strategy = getattr(cfg.trainer, "strategy", None)
    if "_target_" in cfg.strategy:
        log.info(f"Instantiating strategy <{cfg.strategy._target_}>")
        strategy: Strategy = hydra.utils.instantiate(cfg.strategy)
        if "mixed_precision" in strategy.__dict__:
            strategy.mixed_precision.param_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.param_dtype)
                if cfg.strategy.mixed_precision.param_dtype is not None
                else None
            )
            strategy.mixed_precision.reduce_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.reduce_dtype)
                if cfg.strategy.mixed_precision.reduce_dtype is not None
                else None
            )
            strategy.mixed_precision.buffer_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.buffer_dtype)
                if cfg.strategy.mixed_precision.buffer_dtype is not None
                else None
            )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = (
        hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            plugins=plugins,
            strategy=strategy,
        )
        if strategy is not None
        else hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            plugins=plugins,
        )
    )

    if cfg.get("scheduler"):
        if (
            cfg.scheduler.scheduler._target_
            == "flash.core.optimizers.LinearWarmupCosineAnnealingLR"
            and cfg.scheduler.interval == "step"
        ):
            datamodule.setup()  # type: ignore
            num_steps = _num_training_steps(
                datamodule.train_dataloader(), trainer
            )
            log.info(
                f"Setting number of training steps in scheduler to: {num_steps}"
            )
            cfg.scheduler.scheduler.warmup_epochs = (
                num_steps / trainer.max_epochs
            )
            cfg.scheduler.scheduler.max_epochs = num_steps
            log.info(cfg.scheduler)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    log.info(object_dict)
    if cfg.get("train"):
        log.info("Starting training!")
        ckpt_path = None
        if cfg.get("ckpt_path") and os.path.exists(cfg.get("ckpt_path")):
            ckpt_path = cfg.get("ckpt_path")
        elif cfg.get("ckpt_path"):
            log.warning(
                "`ckpt_path` was given, but the path does not exist. Training with new model weights."
            )
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        # if cfg.get("ckpt_path"): 
        #     log.info(f"Loading weights from checkpoint {cfg.ckpt_path}...")
        #     state_dict = torch.load(cfg.ckpt_path)["state_dict"]
        #     encoder_weights = collections.OrderedDict()
        #     for k, v in state_dict.items():
        #         if k.startswith("encoder"):
        #             encoder_weights[k.replace("encoder.", "")] = v
        #     err = model.encoder.load_state_dict(encoder_weights, strict=False)
        #     log.warning(f"Error loading encoder weights: {err}")
        
        #     decoder_weights = collections.OrderedDict()
        #     for k, v in state_dict.items():
        #         if k.startswith("decoder"):
        #     decoder_weights[k.replace("decoder.", "")] = v
        #     err = model.decoder.load_state_dict(decoder_weights, strict=False)
        #     log.warning(f"Error loading decoder weights: {err}")

        splits = ["30", "40", "50", "70", "95"] #fold", "family", "superfamily"]
        wandb_logger = copy.deepcopy(trainer.logger)
        for split in splits:
            dataloader = datamodule.get_test_loader(split)
            trainer.logger = False
            results = trainer.test(
                model=model, dataloaders=dataloader, ckpt_path=cfg.get("ckpt_path") if cfg.get("ckpt_path") else "best"
            )[0]
            results = {f"{k}/{split}": v for k, v in results.items()}
            log.info(f"{split}: {results}")
            wandb_logger.log_metrics(results)
        # trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    src_register_custom_omegaconf_resolvers()
    main()
