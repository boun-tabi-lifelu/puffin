import collections
import os
import pathlib
import rootutils
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
import pandas as pd
import hydra
import lightning as L
import omegaconf
import torch
from loguru import logger as log
from tqdm import tqdm
from proteinworkshop import (
    register_custom_omegaconf_resolvers,
)
from torch_geometric.data import Batch
from graphein.protein.tensor.data import ProteinBatch
from proteinworkshop.models.graph_encoders.esm_embeddings import EvolutionaryScaleModeling


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
from src.utils import (
    extras,
)
from src import (
    register_custom_omegaconf_resolvers as src_register_custom_omegaconf_resolvers,
)


from graphein.protein.tensor.dataloader import ProteinDataLoader
# from proteinworkshop.datasets.base import ProteinDataset
from pathlib import Path 
from loguru import logger 
from graphein.protein.tensor.io import protein_to_pyg
from graphein.protein.utils import (
    download_pdb_multiprocessing,
    get_obsolete_mapping,
)
from loguru import logger
from torch_geometric import transforms as T
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from proteinworkshop.features.sequence_features import amino_acid_one_hot
from proteinworkshop.features.factory import ProteinFeaturiser
from graphein import verbose

verbose(False)
import copy
import torch

from src.data.protein_dataset import ProteinDataset



def embed(cfg: omegaconf.DictConfig):
    # assert cfg.ckpt_path, "No checkpoint path provided."
    assert (
        cfg.output_dir
    ), "No output directory for attributed PDB files provided."



    model_path = Path(cfg.esm_model_path)
    esm_model = EvolutionaryScaleModeling(model_path.parent, model=model_path.name, mlp_post_embed=False, finetune=False)
    for param in esm_model.model.parameters():
        param.requires_grad = False

    esm_model = esm_model.cuda()

    L.seed_everything(cfg.seed)
    if cfg.input_file.endswith('.csv'): 
        df = pd.read_csv(cfg.input_file)
    else: 
        df = pd.read_csv(cfg.input_file, sep="\t", header=None)
        df.columns = ['PDB-chain']
    # data/GeneOntology/nrPDB-GO_test.csv
    df['pdb'] = df['PDB-chain']
    df['chain'] = df['PDB-chain'].apply(lambda s: s.split('-')[1])
    df = df[df['chain'].str.len() == 1]
    # TODO: 
    # FileNotFoundError: 5JM5-A not found in raw directory. Are you sure it's downloaded and has the format pdb?
    
    df = df[df['pdb'] != '5JM5-A'] # test
    df = df[df['pdb'] != '5O61-K'] # train
    df = df[df['pdb'] != '5O61-I'] # train
    df = df[df['pdb'] != '5O61-R'] # train
    df = df[df['pdb'] != '3OHM-B'] # valid
    df = df[df['pdb'] != '2MNT-A'] # valid

    df = df.reset_index()
    
    dataset = ProteinDataset(
            root=cfg.pdb_dir,
            pdb_dir=cfg.pdb_dir,
            pdb_codes=df.pdb, # list(df.pdb if cfg.cluster.file_type == 'pdb' else df.ecod_domain_id),
            # chains=list(df.chain),
            # graph_labels=list(list(df.label)),
            overwrite=False,
            format='pdb',
            in_memory=False
        )

    dataloader = ProteinDataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )

    output_dir = Path(cfg.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for batch in tqdm(dataloader):
        batch = batch.cuda()
        #batch = Batch.from_data_list([batch.cuda()])

        batch_path = Path(cfg.output_dir) / f"{batch.id[0]}.pt"
        embeddings = esm_model.esm_embed(batch).cpu()
        torch.save({'embeddings': embeddings, 'index': batch.residue_id[0]}, batch_path)





# Load hydra config from yaml files and command line arguments.
@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="esm_embed",
)
def main(cfg: omegaconf.DictConfig) -> None:
    """Load and validate the hydra config."""
    extras(cfg)
    embed(cfg)


if __name__ == "__main__":
    
    register_custom_omegaconf_resolvers()
    src_register_custom_omegaconf_resolvers()
    main()
