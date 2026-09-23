from html import unescape

import pytest

from src.gradio_app import (
    UNIT_CLUSTER_ARTIFACT_DIR,
    _build_structure_html,
    _puffin_config,
    _unit_cluster_terms,
    load_uploaded_esm_embeddings,
    parse_pdb_structure,
    select_chain,
)


def test_parse_pdb_structure_reads_residues_and_chains(tmp_path):
    pdb_text = """HEADER    TEST PDB
MODEL        1
ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1      2.000   0.000   0.000  1.00  0.00           C
ATOM      4  N   GLY A   2      3.000   0.000   0.000  1.00  0.00           N
ATOM      5  CA  GLY A   2      4.000   0.000   0.000  1.00  0.00           C
ATOM      6  C   GLY A   2      5.000   0.000   0.000  1.00  0.00           C
ENDMDL
"""
    pdb_path = tmp_path / "demo.pdb"
    pdb_path.write_text(pdb_text)

    structure = parse_pdb_structure(pdb_path)

    assert len(structure["chains"]) == 1
    assert structure["residue_count"] == 2
    assert structure["chain_ids"] == ["A"]


def test_select_chain_defaults_to_first_and_rejects_unknown_chain():
    structure = {"chain_ids": ["B", "A"], "protein_chains": ["A"]}

    assert select_chain(structure) == "A"
    assert select_chain(structure, "A") == "A"
    with pytest.raises(ValueError, match="Chain 'C' was not found"):
        select_chain(structure, "C")


def test_uploaded_esm_embeddings_align_by_residue_id(tmp_path):
    import torch

    embedding_path = tmp_path / "embeddings.pt"
    embeddings = torch.stack(
        [torch.full((1280,), 2.0), torch.full((1280,), 1.0)]
    )
    torch.save(
        {"embeddings": embeddings, "index": ["A:GLY:2", "A:ALA:1"]},
        embedding_path,
    )

    aligned, info = load_uploaded_esm_embeddings(
        embedding_path,
        ["A:ALA:1", "A:GLY:2"],
    )

    assert aligned.shape == (2, 1280)
    assert aligned[0, 0].item() == 1.0
    assert aligned[1, 0].item() == 2.0
    assert info["source"] == "uploaded"
    assert info["alignment"] == "residue_id"


def test_structure_display_hides_or_grays_non_selected_chains(tmp_path):
    pdb_path = tmp_path / "two-chains.pdb"
    pdb_path.write_text(
        """ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY B   1       4.000   5.000   6.000  1.00  0.00           C
HETATM    3  C1  LIG A 101       7.000   8.000   9.000  1.00  0.00           C
END
"""
    )
    payload = {
        "structure": {
            "path": str(pdb_path),
            "name": pdb_path.name,
            "chain_ids": ["A", "B"],
            "residue_count": 2,
            "residue_numbers": [1, 1],
        },
        "chain": "A",
        "assignments": [{"chain": "A", "residue": 1, "unit_id": 2}],
        "units": [{"unit_id": 2, "residue_count": 1}],
        "unit_clusters": [],
    }

    selected_only = unescape(
        _build_structure_html(payload, "Units", "Selected chain only")
    )
    full_pdb = unescape(
        _build_structure_html(payload, "Units", "Full PDB (other chains gray)")
    )

    assert 'viewer.setStyle({"chain": "A"}, {cartoon: {color: \'#B0B0B0\'}});' in selected_only
    assert "viewer.setStyle({}, {cartoon: {color: '#B0B0B0'}});" in full_pdb
    assert "viewer.setStyle({}, {});" in selected_only
    assert "GLY B" not in selected_only
    assert "LIG A" not in selected_only
    assert "GLY B" in full_pdb


def test_ui_uses_bundled_artifact_and_zero_esm_by_default():
    assert (UNIT_CLUSTER_ARTIFACT_DIR / "centroids.npy").is_file()
    assert (UNIT_CLUSTER_ARTIFACT_DIR / "debias_transform.json").is_file()
    assert (UNIT_CLUSTER_ARTIFACT_DIR / "unit_clusters.json").is_file()
    assert len(_unit_cluster_terms()) == 1024

    cfg = _puffin_config("/tmp/model.ckpt")
    assert cfg.encoder.esm_model_path is None

    computed_cfg = _puffin_config("/tmp/model.ckpt", "/tmp/ESM-1b")
    assert computed_cfg.encoder.esm_model_path == "/tmp/ESM-1b"
