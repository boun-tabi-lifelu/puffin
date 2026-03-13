import pandas as pd
import logging
import gzip 
import shutil
import requests

from pathlib import Path 
from typing import List
from src.data.manager import DatasetManager

log = logging.getLogger(__name__)

class ECODManager(DatasetManager):
    def __init__(self, filepath_dir: str="data/", version=292):
        """
        Initialize the ECOD Manager
        """
        self.filepath = Path(filepath_dir) / f"ecod.develop{version}.domains.txt"
        self.data = None
        self.version = version
        self.download()
        self.process()

    def download(self) -> None:
        if not self.filepath.exists():
            url = f"http://prodata.swmed.edu/ecod/distributions/ecod.develop{self.version}.domains.txt"
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

    def read_data(self, filepath: str) -> pd.DataFrame:
        """
        Reads ECOD data from a tab-separated file, skipping comment lines, and returns a DataFrame.
        """
        log.info(f"Reading ECOD data from {filepath}")
        
        # Define the column names based on the file format
        column_names = [
            "source_id",
            "domain_id",
            "d_type",
            "t_id",
            "f_id",
            "ecod_uid",
            "range",
            "representative_status"
        ]
        
        # Read the data while skipping comment lines (lines starting with '#')
        data = pd.read_csv(
            filepath,
            sep=' ',
            header=None,
            names=column_names,
            comment='#'
        )
        
        log.info(f"Successfully read {len(data)} entries from the dataset.")
        return data


    def process(self):
        """
        Processes the ECOD dataset and stores it in the `self.data` attribute.
        """
        log.info("Processing ECOD dataset")
        self.data = self.read_data(self.filepath)
        self.data['chain_id'] = self.data.apply(lambda x: x['range'].split(':')[0] if x['d_type'] == 'PDB' else '', axis=1)

    def get_annotations(self, pdb_ids: List[str], chain_ids: List[str] = None) -> pd.DataFrame:
        """
        Retrieve protein entries from ECOD based on PDB ID and chain.
        
        Parameters:
        pdb_ids (List[str]): List of PDB IDs to search for.
        chain_ids (List[str], optional): List of chain IDs to filter by.
        
        Returns:
        pd.DataFrame: Filtered DataFrame with matching entries.
        """
        pdb_ids = [pdb.lower() for pdb in pdb_ids]
        
        if chain_ids is None:
            protein = self.data[self.data['source_id'].str.lower().isin(pdb_ids)]
        else:
            chain_ids = [chain_id for chain_id in chain_ids]
            protein = self.data[
                (self.data['source_id'].str.lower().isin(pdb_ids)) &
                (self.data['chain_id'].isin(chain_ids))
            ]
        
        if protein.empty:
            raise ValueError(f"No proteins found in ECOD for PDB IDs: {pdb_ids}")
        
        return protein


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize ECOD Manager
    ecod_handler = ECODManager(filepath_dir="data/", version=292)
    
    # Retrieve a specific protein entry
    protein = ecod_handler.get_annotations(pdb_ids=["1kr7"], chain_ids=["A"])
    print(protein)