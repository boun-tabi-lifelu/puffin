from pathlib import Path
import requests
import pandas as pd
import logging
from typing import List
from src.data.manager import DatasetManager 

log = logging.getLogger(__name__)

class SCOPeManager(DatasetManager):
    def __init__(self, filepath: str):
        """
        Initialize the SCOPe handler with the file path for a specific redundancy dataset.
        :param filepath: Path to the SCOPe dataset file.
        """
        self.filepath = Path(filepath)
        self.data = None
        self.download()
        self.process()

    def download(self):
        """
        Download the SCOPe dataset if it does not already exist.
        """
        if not self.filepath.exists():
            url = f"https://scop.berkeley.edu/downloads/scopeseq-2.08/{self.filepath.name}"
            log.info(f"Downloading dataset from {url}")
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
                with self.filepath.open('wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                log.info(f"Dataset downloaded and saved to {self.filepath}")
            else:
                log.error("Failed to download the dataset.")
                raise Exception("Dataset download failed.")

    def read_data(self) -> pd.DataFrame:
        """
        Reads SCOPe data from a fasta file and returns a DataFrame.
        Handles cases where '>' appears in the description.
        """
        log.info(f"Reading SCOPe data from {self.filepath}")
        data = []

        with self.filepath.open('r') as file:
            current_header = None
            current_sequence = []

            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    # If there's a current entry, save it
                    if current_header and current_sequence:
                        identifier, description = current_header.split(maxsplit=1)
                        data.append([identifier, description, ''.join(current_sequence)])
                    
                    # Start a new entry
                    current_header = line[1:]  # Exclude the '>' character
                    current_sequence = []
                else:
                    # Add sequence lines
                    current_sequence.append(line)

            # Add the last entry
            if current_header and current_sequence:
                identifier, description = current_header.split(maxsplit=1)
                data.append([identifier, description, ''.join(current_sequence)])

        return pd.DataFrame(data, columns=['Identifier', 'Description', 'Sequence'])

    def process(self):
        """
        Process the SCOPe dataset.
        """
        log.info("Processing SCOPe dataset")
        self.data = self.read_data()
        self.data['pdb'], self.data['chain'], self.data['pdb_w_chain'] = zip(
            *self.data['Identifier'].apply(
                lambda x: (x[1:5], x[5], f"{x[1:5]}_{x[5]}" if len(x) >= 6 else '')
            )
        )

    def get_annotations(self, pdb_ids: List[str], chain_ids: List[str] = None) -> pd.DataFrame:
        """
        Retrieve protein entries from SCOPe based on PDB ID and chain.
        :param pdb_ids: PDB IDs of the proteins.
        :param chain_ids: Chain IDs of the proteins (optional).
        :return: DataFrame with the protein entries.
        """

        pdb_ids = [pdb_id.lower() for pdb_id in pdb_ids] 
        if self.data is None:
            raise ValueError("SCOPe dataset has not been processed. Call `process()` first.")
        if chain_ids is None:
            protein = self.data[self.data['pdb'].isin(pdb_ids)]
        else:
            chain_ids = [chain_id.lower() for chain_id in chain_ids] 
            protein = self.data[(self.data['pdb'].isin(pdb_ids)) & (self.data['chain'].isin(chain_ids))]
        if protein.empty:
            raise ValueError(f"No proteins found in SCOPe for given ids")
        return protein

class SCOPe40Manager(SCOPeManager):
    def __init__(self):
        super().__init__("data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa.txt")

class SCOPe95Manager(SCOPeManager):
    def __init__(self):
        super().__init__("data/astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa.txt")

if __name__ == "__main__":
    # Initialize handlers for SCOPe 40% and 95% datasets
    scope_40_handler = SCOPeManager(filepath="data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa.txt")
    scope_95_handler = SCOPeManager(filepath="data/astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa.txt")

    # Retrieve a specific protein from SCOPe 40%
    protein_40 = scope_40_handler.get_annotations(pdb_ids=["1kr7"], chain_ids=["A"])
    print(protein_40)

    # Retrieve a specific protein from SCOPe 95%
    protein_95 = scope_95_handler.get_annotations(pdb_ids=["1kr7"], chain_ids=["A"])
    print(protein_95)
