import os
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional

import omegaconf
import pandas as pd
import torch
import wget
from graphein.protein.tensor.data import Protein
#from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger as log
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from proteinworkshop.models.graph_encoders.esm_embeddings import EvolutionaryScaleModeling

from proteinworkshop.datasets.base import ProteinDataModule #, ProteinDataset
from src.data.protein_dataset import ProteinDataset, ProteinDataLoader

# from src.cluster import ProteinDataset
LABEL_LINE: Dict[str, int] = {
    "MF": 1,
    "BP": 5,
    "CC": 9,
}

class GeneOntologyDataset(ProteinDataModule):
    """

    Statistics (test_cutoff=0.95):
        - #Train: 27,496
        - #Valid: 3,053
        - #Test: 2,991

    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        split: str = "BP",
        obsolete="drop",
        pdb_dir: Optional[str] = None,
        format: Literal["mmtf", "pdb"] = "pdb",
        in_memory: bool = False, # Whether to load data into memory, defaults to False.
        dataset_fraction: float = 1.0,
        shuffle_labels: bool = False,
        pin_memory: bool = False,
        num_workers: int = 16,
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False, # Whether to overwrite existing files, defaults to
        esm_model_path: str = '/cta/share/users/esm/ESM-1b',
        esm_embedding_dir: str = None, 
    ) -> None:
        super().__init__()
        self.pdb_dir = pdb_dir
        self.data_dir = Path(path)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.test_dataset_names = ["30", "40", "50", "70", "95"]
        self.dataset_fraction = dataset_fraction
        self.split = split
        self.obsolete = obsolete
        self.format = format
        self.ground_truth = self.data_dir / "ground_truth" / f"{split}_testing.tsv"
        self.in_memory = in_memory
        self.overwrite = overwrite

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.prepare_data_per_node = True
        self.esm_embedding_dir = esm_embedding_dir


        self.shuffle_labels = shuffle_labels

        if transforms is not None:
            self.transform = self.compose_transforms(
                omegaconf.OmegaConf.to_container(transforms, resolve=True)
            )
        else:
            self.transform = None

        self.train_fname = self.data_dir / "nrPDB-GO_train.txt"
        self.val_fname = self.data_dir / "nrPDB-GO_valid.txt"
        self.test_fname = self.data_dir / "nrPDB-GO_test.csv"
        self.label_fname = self.data_dir / "nrPDB-GO_annot.tsv"
        self.url = "https://zenodo.org/record/6622158/files/GeneOntology.zip"

        log.info(
            f"Setting up Gene Ontology dataset. Fraction {self.dataset_fraction}"
        )
        # model_path = Path(esm_model_path)
        # self.esm_model = EvolutionaryScaleModeling(model_path.parent, model=model_path.name, mlp_post_embed=False, finetune=False)
        # for param in self.esm_model.model.parameters():
        #     param.requires_grad = False

        # self.esm_model = self.esm_model.cuda()
    @lru_cache
    def parse_labels(self) -> (Dict[str, torch.Tensor], Dict[str, Iterable[str]]):
        """
        Parse the GO labels from the nrPDB-GO_annot.tsv file.
        """

        log.info(
            f"Loading GO labels for task {self.split} from file {self.label_fname}."
        )

        try:
            label_line = LABEL_LINE[self.split]
        except KeyError as e:
            raise ValueError(f"Task {self.split} not recognised.") from e

        # Load list of all labels
        with open(self.label_fname, "r") as f:
            all_labels = f.readlines()[label_line].strip("\n").split("\t")
        log.info(f"Found {len(all_labels)} labels for task {self.split}.")

        # Load labels for each PDB example
        df = pd.read_csv(self.label_fname, sep="\t", skiprows=12)
        print(df)
        df.columns = ["PDB", "MF", "BP", "CC"]
        df.set_index("PDB", inplace=True)

        # Remove rows with no labels for this task
        labels = df[self.split].dropna().to_dict()
        log.info(f"Found {len(labels)} examples for task {self.split}.")

        # Split GO terms string into list of individual terms
        labels = {k: v.split(",") for k, v in labels.items()}

        # Encode labels into numeric values
        log.info("Encoding labels...")
        label_encoder = LabelEncoder().fit(all_labels)
        
        all_label_ids = label_encoder.transform(all_labels)
        self.label2id = {k: v for k, v in zip(all_labels, all_label_ids)}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        encoded_labels = {
            k: torch.tensor(label_encoder.transform(v))
            for k, v in tqdm(labels.items())
        }
        log.info(f"Encoded {len(labels)} labels for task {self.split}.")
        return encoded_labels, labels

    def _get_dataset(
            self, split: Literal["training", "validation", "testing"], threshold: Literal["30", "40", "50", "70", "95"]=None) -> ProteinDataset:
        df = self.parse_dataset(split, threshold)
        log.info("Initialising Graphein dataset...")
        log.info(
            f"Data dir {self.data_dir} {self.pdb_dir}"
        )
        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=str(self.pdb_dir),
            pdb_codes=list(df.pdb),
            chains=list(df.chain),
            graph_labels=list(list(df.label)),
            overwrite=self.overwrite,
            transform=self.labeller
            if self.transform is None
            else self.compose_transforms([self.labeller] + [self.transform]),
            format=self.format,
            in_memory=self.in_memory,
            esm_embedding_dir=self.esm_embedding_dir
        )

    def train_dataset(self) -> ProteinDataset:
        return self._get_dataset("training")

    def val_dataset(self) -> ProteinDataset:
        return self._get_dataset("validation")

    def test_dataset(self, threshold: Literal["30", "40", "50", "70", "95"]="95") -> ProteinDataset:
        return self._get_dataset("testing", threshold)

    def train_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        return ProteinDataLoader(
            self.test_dataset("95"),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        
    def get_test_loader(self, threshold: Literal["30", "40", "50", "70", "95"]="95") -> ProteinDataLoader:
        return ProteinDataLoader(
            self.test_dataset(threshold),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        
    def download(self):
        if not all(
            os.path.exists(f)
            for f in [
                self.train_fname,
                self.val_fname,
                self.test_fname,
                self.label_fname,
            ]
        ):
            log.info("Downloading dataset...")
            wget.download(self.url, out=str(self.data_dir))
            with zipfile.ZipFile(self.data_dir / "GeneOntology.zip") as f:
                f.extractall(self.data_dir.parent)
        else:
            log.info(f"Found dataset at {self.data_dir}")

    def exclude_pdbs(self):
        pass

    def parse_dataset(
        self, split: Literal["training", "validation", "testing"], 
            threshold: Literal["30", "40", "50", "70", "95"]=None) -> pd.DataFrame:
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches, switch
        """
        Parses the raw dataset files to Pandas DataFrames.
        Maps classes to numerical values.
        """
        # Load ID: label mapping
        class_map, _ = self.parse_labels()
        # Read in IDs of structures in split
        if split == "training":
            data = pd.read_csv(self.train_fname, sep="\t", header=None)
            data = data.sample(frac=self.dataset_fraction)
        elif split == "validation":
            data = pd.read_csv(self.val_fname, sep="\t", header=None)
        elif split == "testing":
            data = pd.read_csv(self.test_fname, sep=",")
            data = data[data[f'<{threshold}%'] == 1]
            data = data[['PDB-chain']].to_csv('tmp.csv', index=False, header=False)
            data = pd.read_csv('tmp.csv', header=None)
 
        else:
            raise ValueError(f"Unknown split: {split}")

        log.info(f"Found {len(data)} original examples in {split}")
        log.info("Removing unlabelled proteins for this task...")
        data = data.loc[data[0].isin(class_map.keys())]
        log.info(f"Found {len(data)} labelled examples in {split}")

        # Map labels to IDs in dataframe
        log.info("Mapping labels to IDs...")
        data["label"] = data[0].map(class_map)
        data.columns = ["pdb", "label"]

        to_drop = ["5EXC-I"] 
        data = data.loc[~data["pdb"].isin(to_drop)]

        data["chain"] = data["pdb"].str[5:]
        data["pdb"] = data["pdb"].str[:4].str.lower()

        if self.obsolete == "drop":
            log.info("Dropping obsolete PDBs")
            data = data.loc[
                ~data["pdb"].str.lower().isin(self.obsolete_pdbs.keys())
            ]
            log.info(
                f"Found {len(data)} examples in {split} after dropping obsolete PDBs"
            )
        else:
            raise NotImplementedError(
                "Obsolete PDB replacement not implemented"
            )
        # logger.info(f"Identified {len(data['label'].unique())} classes in this split: {split}")

        if self.shuffle_labels:
            log.info("Shuffling labels. Expecting random performance.")
            data["label"] = data["label"].sample(frac=1).values
      
        # logger.info(f"Found {len(data)} examples in {split} after removing nonstandard proteins")
        self.labeller = GOLabeller(data)
        log.info(data)
        return data.sample(frac=1)  # Shuffle dataset for batches


class GOLabeller:
    """
    This labeller applies the graph labels to each example as a transform.

    This is required as chains can be used across tasks (e.g. CC, BP or MF) with
    different labels.
    """

    def __init__(self, label_df: pd.DataFrame):
        self.labels = label_df

    def __call__(self, data: Protein) -> Protein:
        pdb, chain = data.id.split("_")
        label = self.labels.loc[
            (self.labels.pdb.str.lower() == pdb.lower()) & (self.labels.chain == chain)
        ].label.item()
        data.graph_y = label
        return data


if __name__ == "__main__":
    import pathlib

    import hydra

    from proteinworkshop import constants

    log.info("Imported libs")
    cfg = omegaconf.OmegaConf.load(
        # constants.SRC_PATH / "config" / "dataset" / "go-bp.yaml"
        Path("configs") / "dataset" / "go-bp.yaml"
    )
    # cfg = omegaconf.OmegaConf.load(constants.SRC_PATH / "config" / "dataset" / "go-mf.yaml")
    # cfg = omegaconf.OmegaConf.load(constants.SRC_PATH / "config" / "dataset" / "go-bp.yaml")
    cfg.datamodule.path = pathlib.Path(constants.DATA_PATH) / "GeneOntology"
    cfg.datamodule.pdb_dir = pathlib.Path(constants.DATA_PATH) / "pdb"
    cfg.datamodule.num_workers = 1
    cfg.datamodule.transforms = []
    log.info("Loaded config")

    ds = hydra.utils.instantiate(cfg)
    ds.datamodule.setup()
    dl = ds["datamodule"].train_dataloader()
    dl = ds["datamodule"].val_dataloader()
    dl = ds["datamodule"].test_dataloader()