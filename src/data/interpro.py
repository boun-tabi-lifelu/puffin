import json
import logging
import pandas as pd 
import multiprocessing
import requests
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, List

from src.data.manager import DatasetManager
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class InterProManager(DatasetManager):
    def __init__(self, pdb_list: List, filepath):
        """
        Initialize the dataset, optionally loading data from a file.
        
        :param json_path: Path to a pre-existing JSON file with annotations.
        :param filepath: Default file to save fetched annotations.
        """
        self.data = None
        self.pdb_list = pdb_list
        self.filepath = Path(filepath)
        self.download()
        self.process()
        

    def download_annotations(self, pdb_id: str) -> Optional[Dict]:
        """
        Fetch annotations for a single PDB ID using the InterPro API.
        
        :param pdb_id: The PDB ID to fetch annotations for.
        :return: JSON response as a dictionary if successful, else None.
        """
        url = f"https://www.ebi.ac.uk/interpro/api/entry/InterPro/structure/PDB/{pdb_id.lower()}/"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                log.error(f"Could not decode JSON for PDB ID {pdb_id}")
        else:
            log.error(f"Failed to retrieve annotations for PDB ID {pdb_id} (Status: {response.status_code})")
        return None

    def download_annotations_worker(self, pdb_id):
        """
        Worker function to download annotations for a single PDB ID.
        """
        try:
            annotations = self.download_annotations(pdb_id)
            return pdb_id, annotations
        except Exception as e:
            log.error(f"Failed to download annotations for {pdb_id}: {e}")
            return pdb_id, None

    def download(self):
        """
        Download and save annotations for a list of PDB IDs using multiprocessing.
        """
        if not self.filepath.exists():
            log.info("Starting download of InterPro annotations with multiprocessing.")
            all_annotations = {}

            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(self.download_annotations_worker, self.pdb_list),
                        desc="Downloading InterPro annotations",
                        total=len(self.pdb_list),
                    )
                )

            # Collect results into a dictionary
            for pdb_id, annotations in results:
                if annotations:
                    all_annotations[pdb_id] = annotations

            if len(all_annotations) > 0: 
                # Save the annotations to a file
                with self.filepath.open("w") as json_file:
                    json.dump(all_annotations, json_file, indent=4)
                log.info(f"Annotations saved to {self.filepath}")
            else: 
                log.info('No annotation is retrieved.')


    # def download(self):
    #     """
    #     Download and save annotations for a list of PDB IDs.
        
    #     :param pdb_ids: List of PDB IDs to download annotations for.
    #     """
    #     if not self.filepath.exists():
    #         all_annotations = {}
    #         for pdb_id in tqdm(self.pdb_list, desc="Downloading InterPro annotations"):
    #             annotations = self.download_annotations(pdb_id)
    #             if annotations:
    #                 all_annotations[pdb_id] = annotations

    #         with open(self.filepath, "w") as json_file:
    #             json.dump(all_annotations, json_file, indent=4)
    #         log.info(f"Annotations saved to {self.filepath}")

    def read_data(self):
        """
        Load annotations from the default annotations file.
        """
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            log.info(f"Annotations loaded from {self.filepath}")
            return data
        else:
            raise FileNotFoundError(f"Annotations file not found: {self.filepath}")

    def get_annotations(self, pdb_ids: List[str], chain_ids: List[str] = None):
        """
        Extracts data for specific PDB IDs.
        
        :param pdb_ids: Set of target PDB IDs to extract information for.
        :param chain_ids: Set of chains 
        :return: Pandas DataFrame containing the filtered information.
        """
        if not self.data:
            raise ValueError("Dataset is not loaded.")
        
        data_rows = []
        for pdb_id in self.data:
            if pdb_ids and pdb_id not in pdb_ids:
                continue
            
            for result in self.data[pdb_id]['results']:
                annotation_type = result['metadata']['type']
                annotation_accession = result['metadata']['accession']
                annotation_name = result['metadata']['name']

                for structure in result['structures']:
                    chain = structure['chain']
                    sequence = structure['sequence']

                    if (structure['entry_structure_locations'] is None or
                        structure['structure_protein_locations'] is None):
                        continue
                    for protein in structure['structure_protein_locations']:
                        for location in structure['entry_structure_locations']:
                            for fragment in location['fragments']:
                                fragment_start = fragment['start']
                                fragment_end = fragment['end']
                                protein_start = protein['fragments'][0]['protein_start']
                                protein_end = protein['fragments'][0]['protein_end']

                                data_rows.append({
                                    'pdb': pdb_id,
                                    'annotation_type': annotation_type,
                                    'annotation_accession': annotation_accession,
                                    'annotation_name': annotation_name,
                                    'chain': chain,
                                    'fragment_start': fragment_start,
                                    'fragment_end': fragment_end,
                                    'fragment_length': fragment_end - fragment_start,
                                    'sequence': sequence,
                                    'protein_start': protein_start,
                                    'protein_end': protein_end
                                })
        df = pd.DataFrame(data_rows)
        df['fragment_range'] = df.apply(lambda r: (r['fragment_start'], r['fragment_end']), axis=1)
        df['id'] = df.apply(lambda r: f"{r['pdb']}-{r['chain']}", axis=1)
        
        if chain_ids is None: 
            protein = df[df['pdb'].isin(pdb_ids)]
        else: 
            chain_ids = [chain_id.lower() for chain_id in chain_ids]
            protein = df[(df['pdb'].isin(pdb_ids)) & (df['chain'].str.lower().isin(chain_ids))]
        if protein.empty:
            raise ValueError(f"No proteins found in BioLiP")
        return protein

    def process(self):
        self.data = self.read_data()
        # self.data = self.get_annotations(self.pdb_list)


if __name__ == '__main__':

    df = pd.read_csv('data/GeneOntology/nrPDB-GO_test.csv')
    df['pdb'], df['chain'] = zip(*df['PDB-chain'].str.split('-'))

    pdb_ids = df['pdb'].tolist()
    interpro_dataset = InterProManager(pdb_list=pdb_ids, filepath="data/GeneOntology/test_interpro.json")
    data = interpro_dataset.get_annotations(pdb_ids)
    print(data)
