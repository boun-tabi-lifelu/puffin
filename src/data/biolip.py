import pandas as pd
import logging
import gzip 
import shutil
import requests

from pathlib import Path 
from typing import List
from src.data.manager import DatasetManager

log = logging.getLogger(__name__)

class BioLiPManager(DatasetManager):
    def __init__(self, filepath: str="data/BioLiP/BioLiP.txt"):
        """
        Initialize the BioLiP Manager
        """
        self.filepath = Path(filepath)
        self.data = None
        self.download()
        self.process()

    def download(self) -> None:
        if not self.filepath.exists():
            # https://zhanggroup.org/BioLiP/download/BioLiP.txt.gz
            url = f"https://zhanggroup.org/BioLiP/download/{self.filepath.name}.gz"
            log.info(f"Downloading dataset from {url}")
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                # Ensure the parent directory exists
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # Temporary .gz file path
                gz_filepath = self.filepath.with_suffix(self.filepath.suffix + '.gz')
                
                with gz_filepath.open('wb') as gz_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        gz_file.write(chunk)
                log.info(f"Compressed dataset downloaded to {gz_filepath}")
                
                # Extract the .gz file
                with gzip.open(gz_filepath, 'rb') as gz_file:
                    with self.filepath.open('wb') as extracted_file:
                        shutil.copyfileobj(gz_file, extracted_file)
                log.info(f"Dataset extracted and saved to {self.filepath}")
                
                # Remove the .gz file after extraction
                gz_filepath.unlink()
                log.info(f"Removed temporary compressed file: {gz_filepath}")

            else:
                log.error("Failed to download the dataset.")
                raise Exception("Dataset download failed.")


    def read_data(self, filepath: str) -> pd.DataFrame:
        """
        Reads BioLiP data from a tab-separated file and returns a DataFrame.
        """
        log.info(f"Reading BioLiP data from {filepath}")
        data = pd.read_csv(
            filepath,
            sep='\t',
            header=None,
            names=[
                'pdb', 'chain', 'resolution', 'binding_site_number_code', 'ligand_id',
                'ligand_chain', 'ligand_serial_number', 'binding_site_residues',
                'binding_site_residues_renumbered', 'catalytic_site_residues',
                'catalytic_site_residues_renumbered', 'ec_number', 'go_terms',
                'binding_affinity_original', 'binding_affinity_binding_moad',
                'binding_affinity_pdbbind_cn', 'binding_affinity_bindingdb',
                'uniprot_id', 'pubmed_id', 'ligand_auth_seq_id', 'receptor_sequence'
            ]
        )
        data['pdb_w_chain'] = data['pdb'] + '_' + data['chain']
        return data

    def process(self):
        log.info("Processing BioLiP dataset")
        self.data = self.read_data(self.filepath)
        


    def get_annotations(self, pdb_ids: List[str], chain_ids: List[str] = None) -> pd.DataFrame:
        """
        Retrieve a single protein entry from BioLiP based on PDB ID and chain.
        """
        pdb_ids = [pdb.lower() for pdb in pdb_ids]
        if chain_ids is None: 
            protein = self.data[self.data['pdb'].str.upper().isin(pdb_ids)]
        else: 
            chain_ids = [chain_id for chain_id in chain_ids]
            protein = self.data[(self.data['pdb'].str.lower().isin(pdb_ids)) & (self.data['chain'].isin(chain_ids))]
        if protein.empty:
            raise ValueError(f"No proteins found in BioLiP")
        return protein

class BioLiPNRManager(BioLiPManager):
    def __init__(self, biolip_path = "data/BioLiP/BioLiP_nr.txt"):
        super().__init__(biolip_path)


if __name__ == "__main__":
    # Initialize handlers 
    biolip_handler = BioLiPManager("data/BioLiP/BioLiP.txt")
    biolip_nr_handler = BioLiPManager("data/BioLiP/BioLiP_nr.txt")


    # Retrieve a specific protein
    protein = biolip_handler.get_annotations(pdb_ids=["1kr7"], chain_ids=["A"])
    print(protein)

    # Retrieve a specific protein
    protein_nr = biolip_nr_handler.get_annotations(pdb_ids=["9ja1"], chain_ids=["A"])
    print(protein_nr)
