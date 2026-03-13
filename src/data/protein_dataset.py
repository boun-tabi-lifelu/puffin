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
from torch_geometric.data import Data, Dataset
from pathlib import Path 
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

class ProteinDataset(Dataset):

    def __init__(
        self,
        pdb_codes: List[str],
        root: Optional[str] = None,
        pdb_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
        pdb_paths: Optional[List[str]] = None,
        chains: Optional[List[str]] = None,
        graph_labels: Optional[List[torch.Tensor]] = None,
        node_labels: Optional[List[torch.Tensor]] = None,
        transform: Optional[List[Callable]] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
        overwrite: bool = False,
        format: Literal["mmtf", "pdb", "ent"] = "pdb",
        in_memory: bool = False,
        store_het: bool = False,
        out_names: Optional[List[str]] = None,
        esm_embedding_dir: Optional[str] = 'data/esm/ESM-1b',
    ):
        self.pdb_codes = pdb_codes
        self.pdb_dir = pdb_dir
        self.pdb_paths = pdb_paths
        self.overwrite = overwrite
        self.chains = chains
        self.node_labels = node_labels
        self.graph_labels = graph_labels
        self.format = format
        self.root = root
        self.in_memory = in_memory
        self.store_het = store_het
        self.out_names = out_names
        self.esm_embedding_dir = Path(esm_embedding_dir)
        self._processed_files = []

        # Determine whether to download raw structures
        if not self.overwrite and all(
            os.path.exists(Path(self.root) / "processed" / p)
            for p in self.processed_file_names
        ):
            logger.info(
                "All structures already processed and overwrite=False. Skipping download."
            )
            self._skip_download = True
        else:
            self._skip_download = False

        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.structures = pdb_codes if pdb_codes is not None else pdb_paths
        if self.in_memory:
            logger.info("Reading data into memory")
            self.data = [
                torch.load(pathlib.Path(self.root) / "processed" / f)
                for f in tqdm(self.processed_file_names)
            ]

    def download(self):
        """
        Download structure files not present in the raw directory (``raw_dir``).

        Structures are downloaded from the RCSB PDB using the Graphein
        multiprocessed downloader.

        Structure files are downloaded in ``self.format`` format (``mmtf`` or
        ``pdb``). Downloading files in ``mmtf`` format is strongly recommended
        as it will be both faster and smaller than ``pdb`` format.

        Downloaded files are stored in ``self.raw_dir``.
        """
        if self.format == "ent":  # Skip downloads from ASTRAL
            logger.warning(
                "Downloads in .ent format are assumed to be from ASTRAL. These data should have already been downloaded"
            )
            return
        if self._skip_download:
            logger.info(
                "All structures already processed and overwrite=False. Skipping download."
            )
            return
        if self.pdb_codes is not None:
            to_download = (
                self.pdb_codes
                if self.overwrite
                else [
                    pdb
                    for pdb in self.pdb_codes
                    if not (
                        os.path.exists(
                            Path(self.raw_dir) / f"{pdb}.{self.format}"
                        )
                        or os.path.exists(
                            Path(self.raw_dir) / f"{pdb}.{self.format}.gz"
                        )
                    )
                ]
            )
            to_download = list(set(to_download))
            logger.info(f"Downloading {len(to_download)} structures")
            file_format = (
                self.format[:-3]
                if self.format.endswith(".gz")
                else self.format
            )
            logger.info(f"Downloading {','.join(to_download)}, {self.raw_dir}")

            download_pdb_multiprocessing(
                to_download, self.raw_dir, format=file_format
            )

    def len(self) -> int:
        """Return length of the dataset."""
        return len(self.pdb_codes)

    @property
    def raw_dir(self) -> str:
        """Returns the path to the raw data directory.

        :return: Raw data directory.
        :rtype: str
        """
        return os.path.join(self.root, "raw") if self.pdb_dir is None else self.pdb_dir  # type: ignore

    @property
    def raw_file_names(self) -> List[str]:
        """Returns the raw file names.

        :return: List of raw file names.
        :rtype: List[str]
        """
        if self._skip_download:
            return []
        if self.pdb_paths is None:
            return [f"{pdb}.{format}" for pdb in self.pdb_codes]
        else:
            return list(self.pdb_paths)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """Returns the processed file names.

        This will either be a list in format [``{pdb_code}.pt``] or
        a list of [{pdb_code}_{chain(s)}.pt].

        :return: List of processed file names.
        :rtype: Union[str, List[str], Tuple]
        """
        if self._processed_files:
            return self._processed_files
        if self.overwrite:
            return ["this_forces_a_processing_cycle"]
        if self.out_names is not None:
            return [f"{name}.pt" for name in self.out_names]
        if self.chains is not None:
            return [
                f"{pdb}_{chain}.pt"
                for pdb, chain in zip(self.pdb_codes, self.chains)
            ]
        else:
            return [f"{pdb}.pt" for pdb in self.pdb_codes]

    def load_esm_embeddings(self, protein_id: str) -> torch.Tensor:
        """Loads ESM embeddings for a given protein ID."""
        embedding_path = self.esm_embedding_dir / f"{protein_id}.pt"
        if not embedding_path.exists():
            #logger.info(f"ESM embedding for {protein_id} not found: {embedding_path}")
            return None
        return torch.load(embedding_path)

    def process(self):
        """Process raw data into PyTorch Geometric Data objects with Graphein.

        Processed data are stored in ``self.processed_dir`` as ``.pt`` files.
        """
        if not self.overwrite:
            if self.chains is not None:
                index_pdb_tuples = [
                    (i, pdb)
                    for i, pdb in enumerate(self.pdb_codes)
                    if not os.path.exists(
                        Path(self.processed_dir) / f"{pdb}_{self.chains[i]}.pt"
                    )
                ]
            else:
                index_pdb_tuples = [
                    (i, pdb)
                    for i, pdb in enumerate(self.pdb_codes)
                    if not os.path.exists(
                        Path(self.processed_dir) / f"{pdb}.pt"
                    )
                ]
            logger.info(
                f"Processing {len(index_pdb_tuples)} unprocessed structures"
            )
        else:
            index_pdb_tuples = [
                (i, pdb) for i, pdb in enumerate(self.pdb_codes)
            ]

        raw_dir = Path(self.raw_dir)
        for index_pdb_tuple in tqdm(index_pdb_tuples):
            try:
                (
                    i,
                    pdb,
                ) = index_pdb_tuple  # NOTE: here, we unpack the tuple to get each PDB's original index in `self.pdb_codes`
                path = raw_dir / f"{pdb}.{self.format}"

                if path.exists():
                    path = str(path)
                elif path.with_suffix("." + self.format + ".gz").exists():
                    path = str(path.with_suffix("." + self.format + ".gz"))
                else:
                    raise FileNotFoundError(
                        f"{pdb} not found in raw directory. Are you sure it's downloaded and has the format {self.format}?"
                    )
                graph = protein_to_pyg(
                    path=path,
                    chain_selection=self.chains[i]
                    if self.chains is not None
                    else "all",
                    keep_insertions=True,
                    store_het=self.store_het,
                )
                if self.esm_embedding_dir:
                    emb_data = self.load_esm_embeddings(pdb)
                    graph.esm_embeddings = emb_data["embeddings"]
                    graph.esm_id = emb_data["index"]
            except Exception as e:
                logger.error(f"Error processing {pdb} {self.chains[i]}: {e}")  # type: ignore
                raise e

            if self.out_names is not None:
                fname = self.out_names[i] + ".pt"
            else:
                fname = (
                    f"{pdb}.pt"
                    if self.chains is None
                    else f"{pdb}_{self.chains[i]}.pt"
                )

            graph.id = fname.split(".")[0]

            if self.graph_labels is not None:
                graph.graph_y = self.graph_labels[i]  # type: ignore

            if self.node_labels is not None:
                graph.node_y = self.node_labels[i]  # type: ignore

            torch.save(graph, Path(self.processed_dir) / fname)
            self._processed_files.append(fname)
        logger.info("Completed processing.")

    def get(self, idx: int) -> Data:
        """
        Return PyTorch Geometric Data object for a given index.

        :param idx: Index to retrieve.
        :type idx: int
        :return: PyTorch Geometric Data object.
        """
        if self.in_memory:
            return self._batch_format(copy.deepcopy(self.data[idx]))

        if self.out_names is not None:
            fname = f"{self.out_names[idx]}.pt"
        elif self.chains is not None:
            fname = f"{self.pdb_codes[idx]}_{self.chains[idx]}.pt"
        else:
            fname = f"{self.pdb_codes[idx]}.pt"

        batch = self._batch_format(torch.load(Path(self.processed_dir) / fname))
        # TODO: check fname

        # pdb, chain = fname.replace('.pt', '').split('_')
        # fname = f'{pdb.upper()}-{chain}'

        fname = fname.replace('.pt', '')
        if '_' in fname: 
            fname = fname.replace('_', '-')
            
        data = self.load_esm_embeddings(fname)
        #print(fname, data)
        if data is not None:
             batch.esm_embeddings = data["embeddings"]
             batch.esm_id = data["index"]
        return batch 

    def _batch_format(self, x: Data) -> Data:
        # Set this to ensure proper batching behaviour
        x.x = torch.zeros(x.coords.shape[0])  # type: ignore
        x.amino_acid_one_hot = amino_acid_one_hot(x)
        x.seq_pos = torch.arange(x.coords.shape[0]).unsqueeze(
            -1
        )  # Add sequence position
        x.esm_embeddings = torch.zeros((x.coords.shape[0], 1280))
        x.esm_id = x.residue_id
        return x

from collections.abc import Mapping
from typing import List, Optional, Sequence, Union

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData

from graphein.protein.tensor.data import ProteinBatch


def align_embeddings_with_structure(residue_ids, esm_ids, esm_embeddings):
    """
    Align ESM embeddings to a list of residue_ids by creating a zero tensor
    (with shape = [len(residue_ids), embedding_dim]) and then populating
    it for residues that are found in esm_ids.

    Args:
        residue_ids (list): A list of residue identifiers in the structure order.
        esm_ids (list): A list of residue identifiers in ESM embedding order.
        esm_embeddings (torch.Tensor): A torch Tensor of shape
            (len(esm_ids), embedding_dim) or possibly (len(esm_ids)+1, embedding_dim)
            if there's a special token at index 0.

    Returns:
        aligned_embeddings (torch.Tensor): A zero-initialized tensor of shape
            (len(residue_ids), embedding_dim) with the corresponding ESM embeddings
            placed at their matching residue positions.
        matching_residues (list): A list of residues that exist in both
            residue_ids and esm_ids.
    """
    # Map residue_id -> index for both residue_ids and esm_ids
    residue_dict = {rid: idx for idx, rid in enumerate(residue_ids)}
    esm_dict = {rid: idx for idx, rid in enumerate(esm_ids)}
    #print(residue_ids[:3], esm_ids[:3])
    # Find which residues appear in both
    common_residues = set(residue_dict.keys()).intersection(esm_ids)

    # Initialize the aligned embedding tensor with zeros
    embedding_dim = esm_embeddings.shape[1]
    aligned_embeddings = torch.zeros(
        (len(residue_ids), embedding_dim),
        dtype=esm_embeddings.dtype,
        device=esm_embeddings.device
    )

    # Copy embeddings for residues that appear in both lists
    for res in common_residues:
        aligned_embeddings[residue_dict[res]] = esm_embeddings[esm_dict[res]]

    return aligned_embeddings, list(common_residues)

class Collater:
    def __init__(self, follow_batch, exclude_keys): #, esm_model):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys


    def __call__(self, batch):
        elem = batch[0]
        for elem in batch: 
            if elem.x.shape[0] != elem.esm_embeddings.shape[0]:
                # try: 
                elem.esm_embeddings, elem.esm_id = align_embeddings_with_structure(elem.residue_id, elem.esm_id, elem.esm_embeddings)
                # except: 
                #     print(f"{elem.id}  {elem.x.shape[0]} {earlier_size} {elem.esm_embeddings.shape[0]} {len(elem.residue_id)} {len(elem.esm_id)} {elem.residue_id} {elem.esm_id}") #  {i.residue_id} {i.esm_id}"

            assert elem.x.shape[0] == elem.esm_embeddings.shape[0], f"{elem.id}  {elem.x.shape[0]} {earlier_size} {elem.esm_embeddings.shape[0]} {len(elem.residue_id)} {len(elem.esm_id)} {elem.residue_id} {elem.esm_id}" #  {i.residue_id} {i.esm_id}"
        if isinstance(elem, BaseData):
            return ProteinBatch.from_data_list(
                batch, self.follow_batch, self.exclude_keys
            )
            # protein_batch = ProteinBatch.from_data_list(
            #     batch, self.follow_batch, self.exclude_keys
            # )
            # protein_batch.esm_embeddings = self.esm_model.esm_embed(protein_batch.cuda())
            # return protein_batch.cpu()

        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")

    def collate(self, batch):  # Deprecated...
        return self(batch)


class ProteinDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.


    Data objects can be either of type :class:`~graphein.protein.tensor.data.Protein` or
    :class:`~torch_geometric.data.Data`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        #esm_model = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys), # , esm_model),
            multiprocessing_context='spawn',  
            **kwargs,
        )