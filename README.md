<h1 align="center">
  <img src="puffin.png" height="40" style="vertical-align:-6px;"> PUFFIN: Protein Unit Discovery with Functional Supervision
</h1>


<!--
<p align="center">
  <a href="https://doi.org/10.1093/bioinformatics/xxxxx">
    <img src="https://img.shields.io/badge/DOI-10.1093/bioinformatics/xxxxx-blue">
  </a>
  <img src="https://img.shields.io/badge/Venue-Bioinformatics-blue">
  <img src="https://img.shields.io/badge/Conference-ISMB%202026-orange">
</p>
-->

## Installation

### System requirements

* Linux + NVIDIA GPU (recommended)
* Recent NVIDIA drivers + CUDA toolkit compatible with your PyTorch build (example below uses **CUDA 11.8**)




### Install Miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

Restart your shell or:

```bash
source ~/.bashrc
```



### Create and activate a conda environment

```bash
conda create -n pfp python=3.10 -y
conda activate pfp
```


### Install ProteinWorkshop (dependency)

```bash
git clone https://github.com/a-r-j/ProteinWorkshop
cd ProteinWorkshop
pip install -e .
```

Install PyTorch (example: CUDA 11.8 wheels):

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

Additional utilities:

```bash
pip install rootutils cafaeval
pip install umap-learn pandas matplotlib datashader bokeh holoviews scikit-image colorcet
```



### Configure ProteinWorkshop environment variables

If ProteinWorkshop provides an example env file:

```bash
# If present:
# cp .env.example .env
# otherwise:
touch .env
```

Edit `.env`:

```bash
vi .env
```

Example:

```bash
ROOT_DIR="/home/user/ProteinWorkshop"
RUNS_PATH="/home/user/ProteinWorkshop/runs"
DATA_PATH="/home/user/ProteinWorkshop/data"

WANDB_API_KEY=""          # optional
WANDB_ENTITY="gokceuludogan"  # optional
WANDB_PROJECT="puffin"    # optional
```



## Data

This project relies on the **ProteinWorkshop GO datasets** (GO-MF) and (optionally) InterPro annotations for unit-boundary evaluation.

### Required datasets

* `go-mf`

### Optional (for `src/interpro_proto_go_term_comparison.py`)

* `data/GeneOntology/test_interpro.json` (InterPro annotations)


All datasets are expected to live under the directory specified by:

```bash
DATA_PATH=/path/to/data
```

Make sure this is correctly set in your `.env` file (see **Installation**).

```
data/
└── GeneOntology/
```


### Gene Ontology (GO)

GO datasets are downloaded **via ProteinWorkshop**.

Configure your `.env`:

```bash
DATA_PATH="/path/to/ProteinWorkshop/data"
```

Then run:

```bash
workshop download go-mf
workshop download go-bp
workshop download go-cc
```

This will populate:

```
data/GeneOntology/
├── nrPDB-GO_train.txt
├── nrPDB-GO_valid.txt
├── nrPDB-GO_test.txt
├── nrPDB-GO_annot.tsv
└── ...
```

These datasets are used for:

* GO prediction (MF / BP / CC)
* Clustering evaluation
* Unit discovery benchmarks



#### IA scores
To produce IA scores for PDB with different versions of GO ontology

```bash
python src/data/ia.py --annot .\data\GeneOntology\ground_truth\terms.tsv \
	--graph .\data\go-basic-2020-06-01.obo --prop \
	--outfile data\IA-nrPDB-go-basic-2020-06-01.txt
```


## Experiment Pipeline

1. **Train a model** (PUFFIN / Protygus / mincut variants)
2. **Extract protein units (segments)** from trained models
3. **Characterize units** (size, structure, connectivity, random baselines)
4. **Evaluate unit function** (GO neighborhood tests)
5. **Analyze unit–InterPro correspondence** (optional, deeper analysis)

Each step is modular and can be run independently.


## Training PUFFIN

Training uses **Hydra** (`src/train.py`).  
PUFFIN is typically trained with a **dual objective**:
- protein-level GO prediction
- unit discovery

### Example: train PUFFIN with K=64 units
```bash
python src/train.py \
  name=puffin \
  encoder=puffin \
  encoder.gnn_type=GAT \
  encoder.hidden_dim=512 \
  encoder.num_clusters=64 \
  encoder.num_res_gnn_layers=2 \
  encoder.num_seg_gnn_layers=2 \
  encoder.proj_layer=true \
  encoder.fuse_lm_method=sum \
  objective_type=dual \
  function_weight=1.0 \
  unit_weight=0.5
````

**What this does**

* Trains a PUFFIN model with 64 latent units
* Uses a GAT-based residue and unit GNN
* Optimizes both GO prediction and unit discovery
* Saves checkpoints under `models/puffin/`

---

## Unit extraction

After training, units are extracted using `src/cluster.py`.

### Example: extract PUFFIN segments on the test split

```bash
python src/cluster.py \
  encoder=puffin \
  encoder.num_clusters=64 \
  ckpt_path models/puffin/epoch_*.ckpt \
  cluster.input_file data/GeneOntology/nrPDB-GO_test.txt \
  cluster.split test \
  output_dir units/puffin/test/
```

**Outputs**

* `test_residue_assignments.csv`
* `test_segment_embeddings.npy`
* `test_segment_metadata.csv`

These define residue → unit assignments and unit embeddings.

## Unit characterization (structure & statistics)

This step analyzes extracted units:

* size distributions
* structural compactness
* contact-based coherence
* comparison to size-matched random baselines

### Example: characterize PUFFIN units (test split)

```bash
python src/segment_characterize.py \
  --cluster_dir ismb26/segments/puffin_K64/test \
  --output_dir ismb26/results/segment_reports/puffin_K64/test \
  --prefix test \
  --structure_dir data/pdb_chain \
  --contact_cutoff 10.0 \
  --random_baseline
```


## Unit functional evaluation (GO neighborhoods)

Units are evaluated by checking whether **nearby units in embedding space** share GO functions.

### Example: segment-level GO evaluation

```bash
python src/segment_func_eval.py \
  --segments_root units/ \
  --model_name puffin_K64 \
  --split valid \
  --annotation_dir data/GeneOntology \
  --go_aspect MF \
  --out_root results/unit_func_reports \
  --k_neighbors 50 \
  --n_queries 5000
```


##  Unit ↔ InterPro correspondence 

For deeper biological validation, units can be compared to **InterPro annotations**:

* map InterPro regions to best-matching units (IoU)
* propagate InterPro2GO mappings
* evaluate GO ranking quality (Hit@K, MRR)

This analysis combines:

* unit assignments
* cluster enrichment results
* InterPro JSON + InterPro2GO mappings

* Associated script: `src/interpro_proto_go_term_comparison.py`

---

## Baselines supported

The same pipeline applies to:

* **Mincut pooling** (unsupervised learned units)
* **ESM + k-means** (structure-agnostic baseline)

All baselines produce compatible outputs under:

```
ismb26/segments/<model_name>/<split>/
```

so they can be evaluated identically.

---

## Typical directory layout

```
ismb26/
├── models/                 # trained checkpoints
├── units/               # extracted units
│   └── puffin/
│       ├── train/
│       ├── valid/
│       └── test/
├── results/
│   ├── unit_reports/    # structural characterization
│   ├── unit_func_reports/ # GO neighborhood eval
│   └── func_eval/          # protein-level eval logs
```


## Reference

If you use this repository, please cite the following related [paper]():
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  publisher={}
}
```


## License

This code base is licensed under the MIT license. See [LICENSE](license.md) for details.
