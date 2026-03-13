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
from loguru import logger as log
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from graphein.protein.tensor.dataloader import ProteinDataLoader

from proteinworkshop.datasets.base import ProteinDataModule, ProteinDataset
from src.data.go_datamodule import GOLabeller

LABEL_LINE: Dict[str, int] = {
    "MF": 1,
    "BP": 5,
    "CC": 9,
}


class SingleProteinDataset:

    def __init__(
        self,
        pdb_path: str,
        data_dir: str, 
        split: str = "BP"
    ) -> None:
        super().__init__()
        self.pdb_path = Path(pdb_path)
        self.pdb_dir = self.pdb_path.parent
        self.pdb_id = self.pdb_path.stem
        self.pdb, self.chain = self.pdb_id.split('-')
        self.data_dir = Path(data_dir)
        self.split = split

        self.label_fname = self.data_dir / "nrPDB-GO_annot.tsv"

    @lru_cache
    def parse_labels(self) -> Dict[str, torch.Tensor]:
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
        df.columns = ["PDB", "MF", "BP", "CC"]
        df.set_index("PDB", inplace=True)

        # Remove rows with no labels for this task
        labels = df[self.split].dropna().to_dict()
        
        # Split GO terms string into list of individual terms
        labels = {k: v.split(",") for k, v in labels.items() if k == self.pdb_id}
        
        # Encode labels into numeric values
        log.info("Encoding labels...")
        label_encoder = LabelEncoder().fit(all_labels)
        
        all_label_ids = label_encoder.transform(all_labels)
        self.label2id = {k: v for k, v in zip(all_labels, all_label_ids)}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        terms = labels.copy()
        labels = {
            k: torch.tensor(label_encoder.transform(v))
            for k, v in tqdm(labels.items())
        }
        log.info(f"Encoded {len(labels)} labels for task {self.split}.")
        return labels, terms

    def _get_dataset(
            self ) -> ProteinDataset:
        df = self.parse_dataset()
        log.info("Initialising Graphein dataset...")
        log.info(
            f"Data dir {self.data_dir} {self.pdb_dir}"
        )
        return ProteinDataset(
            root=str(self.data_dir),
            pdb_dir=str(self.pdb_dir),
            pdb_codes=list(df.pdb) if len(df) > 0 else [self.pdb],
            chains=list(df.chain) if len(df) > 0 else [self.chain],
            graph_labels=list(list(df.label)) if len(df) > 0 else [self.chain],
            overwrite=False,
            transform=self.labeller if len(df) > 0 else None,
        )

  
    def parse_dataset(
        self) -> pd.DataFrame:
        # sourcery skip: remove-unnecessary-else, swap-if-else-branches, switch
        """
        Parses the raw dataset files to Pandas DataFrames.
        Maps classes to numerical values.
        """

        self.class_map, self.classes = self.parse_labels()

        data = []
        for key in self.class_map.keys():
            pdb, chain = key.split('-')
            tensor_value = self.class_map[key]  
            data.append({'pdb': pdb, 'chain': chain, 'label': tensor_value})

        df = pd.DataFrame(data)

        self.labeller = GOLabeller(df)
        return df




if __name__ == "__main__":
    
    dataset = SingleProteinDataset(
        pdb_path='/cta/users/guludogan/ProteinWorkshop/proteinworkshop/data/GeneOntology/pdb/2A99-A.pdb', 
        data_dir='/cta/users/guludogan/ProteinWorkshop/proteinworkshop/data/GeneOntology/', 
        split= "BP")


    df = dataset._get_dataset()
    instance = df.get(0)
    print(dataset.classes)
    print(dataset.class_map)
    print(df.get(0))
    print(df.get(0).graph_y)
    instance.graph_y = list(dataset.class_map.values())[0]
    print(instance.graph_y )
    print(dataset.id2label[instance.graph_y[0].item()])
