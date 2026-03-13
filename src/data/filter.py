import os
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional

import omegaconf
import pandas as pd
import torch
from graphein.protein.tensor.data import Protein
from graphein.protein.tensor.dataloader import ProteinDataLoader
from loguru import logger as log
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

LABEL_LINE: Dict[str, int] = {
    "MF": 1,
    "BP": 5,
    "CC": 9,
}


class GeneOntologyDataset:
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
        format: Literal["mmtf", "pdb"] = "mmtf",
        in_memory: bool = False,
        dataset_fraction: float = 1.0,
        shuffle_labels: bool = False,
        pin_memory: bool = True,
        num_workers: int = 16,
        transforms: Optional[Iterable[Callable]] = None,
        overwrite: bool = False,
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

        self.in_memory = in_memory
        self.overwrite = overwrite

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.prepare_data_per_node = True

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

        log.info(
            f"Setting up Gene Ontology dataset. Fraction {self.dataset_fraction}"
        )



    def parse_dataset(
        self, split: Literal["training", "validation", "testing"], threshold: Literal["30", "40", "50", "70", "95"]="95") -> pd.DataFrame:
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches, switch
        """
        Parses the raw dataset files to Pandas DataFrames.
        Maps classes to numerical values.
        """
        # Load ID: label mapping
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
        # labels = {
        #     k: torch.tensor(label_encoder.transform(v))
        #     for k, v in tqdm(labels.items())
        # }
        
        all_labels_id = label_encoder.transform(all_labels)
        pd.DataFrame({'labels': all_labels, 'ids': all_labels_id}).to_csv(self.data_dir / f'{self.split}_mapping.csv', index=False, header=False)
        log.info(f"Encoded {len(labels)} labels for task {self.split}.")
        class_map = labels

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
        data.explode("label").to_csv(self.data_dir / f"ground_truth/{self.split}_{split}.tsv", index=False, sep='\t', header=False)

        data["chain"] = data["pdb"].str[5:]
        data["pdb"] = data["pdb"].str[:4].str.lower()

        if self.shuffle_labels:
            log.info("Shuffling labels. Expecting random performance.")
            data["label"] = data["label"].sample(frac=1).values

        # logger.info(f"Found {len(data)} examples in {split} after removing nonstandard proteins")
        self.labeller = GOLabeller(data)
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
            (self.labels.pdb == pdb) & (self.labels.chain == chain)
        ].label.item()
        data.graph_y = label
        return data


for subontology in ["BP", "MF", "CC"]:
    for split in ["training", "validation", "testing"]:
        dataset = GeneOntologyDataset(
            path='data/GeneOntology',
            pdb_dir='data/pdb',
            format='mmtf', # pdb
            batch_size=32,
            dataset_fraction=1.0,
            shuffle_labels=False,
            pin_memory=True,
            num_workers=8,
            split=subontology,
            transforms=None, # uses GOLabeler by default.
            overwrite=False,
            in_memory=True,
        )

        dataset.parse_dataset(split=split)
