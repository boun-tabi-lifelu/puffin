"""Regression checks for released-checkpoint ESM input compatibility."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from src.data.protein_dataset import ProteinDataset


class ESMFeatureModeTest(unittest.TestCase):
    def make_dataset(self, root, mode="zero", in_memory=False):
        processed = Path(root) / "processed"
        processed.mkdir(exist_ok=True)
        graph = Data(coords=torch.zeros(3, 3),
                     residue_id=["ALA:A:1", "ALA:A:2", "ALA:A:3"],
                     esm_embeddings=torch.ones(3, 1280))
        torch.save(graph, processed / "1abc_A.pt")
        cache = Path(root) / "esm"
        cache.mkdir(exist_ok=True)
        torch.save({"embeddings": torch.ones(3, 1280), "index": graph.residue_id},
                   cache / "1ABC-A.pt")
        return ProteinDataset(pdb_codes=["1abc"], chains=["A"], root=root,
                              pdb_dir=root, esm_embedding_dir=str(cache),
                              esm_features=mode, in_memory=in_memory)

    def test_zero_ignores_cached_and_processed_embeddings(self):
        for in_memory in (False, True):
            with self.subTest(in_memory=in_memory), tempfile.TemporaryDirectory() as root:
                dataset = self.make_dataset(root, in_memory=in_memory)
                with patch("src.data.protein_dataset.amino_acid_one_hot",
                           return_value=torch.zeros(3, 20)), patch.object(
                               Path, "exists", side_effect=AssertionError("cache lookup")):
                    batch = dataset.get(0)
                self.assertEqual(tuple(batch.esm_embeddings.shape), (3, 1280))
                self.assertEqual(torch.count_nonzero(batch.esm_embeddings).item(), 0)
                self.assertEqual(batch.esm_id, batch.residue_id)

    def test_cached_mode_loads_real_features(self):
        for in_memory in (False, True):
            with self.subTest(in_memory=in_memory), tempfile.TemporaryDirectory() as root:
                dataset = self.make_dataset(root, mode="cached", in_memory=in_memory)
                with patch("src.data.protein_dataset.amino_acid_one_hot",
                           return_value=torch.zeros(3, 20)):
                    batch = dataset.get(0)
                self.assertTrue(torch.equal(batch.esm_embeddings, torch.ones(3, 1280)))

    def test_missing_cache_falls_back_to_zeros(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = self.make_dataset(root, mode="cached")
            (Path(root) / "esm" / "1ABC-A.pt").unlink()
            with patch("src.data.protein_dataset.amino_acid_one_hot",
                       return_value=torch.zeros(3, 20)):
                batch = dataset.get(0)
            self.assertEqual(torch.count_nonzero(batch.esm_embeddings).item(), 0)

    def test_invalid_mode_and_missing_cache_directory(self):
        with self.assertRaises(ValueError):
            ProteinDataset(pdb_codes=[], esm_features="invalid")
        with self.assertRaises(ValueError):
            ProteinDataset(pdb_codes=[], esm_features="cached", esm_embedding_dir=None)


if __name__ == "__main__":
    unittest.main()
