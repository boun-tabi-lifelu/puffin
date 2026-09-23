# Gradio PUFFIN protein unit explorer

Run the app locally with:

```bash
conda activate pfp
pip install -e ".[ui]"
python src/gradio_app.py
```

Create and activate the PUFFIN environment as described in the main README
before installing the UI extra. The `scripts/run_gradio_app.sh` launcher uses
the active environment's Python. Set `PUFFIN_PYTHON` to override it.

The web stack is pinned to Gradio 4.44.1, FastAPI 0.103.2, Starlette 0.27.0,
and Pydantic 2.6.4. Newer Pydantic schemas are not compatible with Gradio
4.44.1 API schema generation.

Each analysis is persisted under `results/gradio/<run_id>/` with `input.pdb` and
`results.json`. Set `PUFFIN_RESULTS_DIR` before launching to use another storage
location.

The app accepts a PDB file and produces:
- a left-side input panel,
- a center 3D structure preview,
- a right-side table of PUFFIN-discovered units with downloadable JSON output.

The checkpoint is downloaded from
[`lifelu/puffin`](https://huggingface.co/lifelu/puffin/tree/main) as
`model.ckpt`. The UI extracts the selected chain into a canonical
`<pdb_id>-<chain>.pdb` file and builds the graph with the public-release
single-PDB inference path.

Zero ESM features remain the default to reproduce the historical ISMB26
test-set behavior and matching K1024 unit-cluster assignments. The UI also
supports uploading precomputed ESM-1b embeddings or computing ESM-1b features
at inference time. Uploaded files may be tensors with shape `[N, 1280]`, or the
dictionary produced by `src/esm_embed.py` with `embeddings` and `index` fields.

`Compute ESM` uses the model/cache identifier from `PUFFIN_ESM_MODEL_PATH`,
which defaults to `~/.cache/puffin/ESM-1b`. ProteinWorkshop downloads missing
ESM-1b weights into that cache on first use. Checkpoint selection remains out
of the UI and can be overridden with `PUFFIN_CHECKPOINT_PATH` before launch.

Global unit-cluster assignments and GO associations are loaded from the bundled
`artifacts/puffin-unit-cluster-functions` directory. No shared filesystem
paths are required.

The structure panel can show only the selected chain or the full uploaded PDB.
In full-PDB mode, non-selected chains are displayed in gray while the selected
chain retains its PUFFIN unit or unit-cluster colors.
