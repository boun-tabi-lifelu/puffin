import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class DatasetManager(ABC):
    def __init__(self, filepath: str):
        """
        Abstract base class for managing datasets.
        :param filepath: Path to the dataset file.
        """
        self.filepath = filepath

    @abstractmethod
    def download(self):
        """Download the dataset if it does not exist."""
        pass

    @abstractmethod
    def read_data(self):
        """Read the dataset and return it as a DataFrame."""
        pass

    @abstractmethod
    def get_annotations(self, pdb_ids, pdb_chains=None):
        """Retrieve specific entries based on PDB IDs and chain IDs."""
        pass

