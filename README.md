# iLP SMILES Preparation Tool

iLP prepares input SMILES for the trained iLP models. It canonicalizes and desalts molecules, enumerates protonation states, embeds each protonation-state SMILES with the MolAI encoder, and uses a five-model iLP ensemble to select the highest-scoring prepared SMILES.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `iLP_run.py` | Main CLI pipeline runner. |
| `Dockerfile` | GPU-capable runtime image based on TensorFlow 2.7.1. |
| `requirements.txt` | Python packages installed on top of the TensorFlow image. |
| `MolAI/` | MolAI SMILES encoder and character dictionaries. |
| `iLP/` | Five trained iLP ensemble models. |
| `examples/smoke_input.csv` | Small input file for container smoke testing. |
| `tests/` | Lightweight regression tests for group-safe batching. |

## Citation

This tool uses the MolAI SMILES encoder. If you use this repository, please cite the related MolAI paper:

Mahdizadeh, S. J.; Eriksson, L. A. MolAI: A Deep Learning Framework for Data-Driven Molecular Descriptor Generation and Advanced Drug Discovery Applications. *Journal of Chemical Information and Modeling* **2025**, *65* (19), 9892-9909. https://doi.org/10.1021/acs.jcim.5c00491

## Pipeline

1. `input.csv` -> `SMILES_canonical.csv`
   - Reads a CSV with `smiles,id` columns.
   - Uses RDKit to parse, desalt, and canonicalize each SMILES.

2. `SMILES_canonical.csv` -> `dimorph_canonical.csv`
   - Uses Dimorphite-DL-compatible protonation enumeration.
   - Default pH window is `5.0` to `9.0`.
   - Default maximum variants per molecule is `265`.

3. `dimorph_canonical.csv` -> `prepped_SMILES.csv`
   - Normalizes `Cl`/`Br` to model tokens `X`/`Y`.
   - Filters to the MolAI character set and model length limit.
   - Encodes each valid protonation state with `MolAI/smi2lat_epoch_6.h5`.
   - Scores each latent vector with `iLP/model_1_1.h5` through `model_1_5.h5`.
   - Averages the five model probabilities.
   - Keeps one highest-scoring protonation state per molecule group. Exact ties are resolved by keeping the first max-scoring variant unless `--tie-policy all` is explicitly requested.

The runner batches model inference for memory efficiency, but prediction batches are built from complete protonation groups. This prevents a molecule from being split across two batches before the max-probability selection is applied.

The pipeline also writes `discarded_molecules.csv`. This report has one row per input molecule that fails to produce a final prepared SMILES row, with the failing stage and reason.

## Build the Docker Image

```bash
docker build -t ilp:latest .
```

The image includes the model artifacts from `MolAI/` and `iLP/`, so runtime inputs can be mounted separately.

The image uses a GPU automatically when Docker exposes one with `--gpus all`. If no compatible GPU is visible to TensorFlow, the tool continues on CPU. Add `--require-gpu` only when you want the run to fail instead of falling back.

## Smoke Test

PowerShell:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
docker run --rm `
  --gpus all `
  -v "${PWD}\outputs:/work" `
  ilp:latest `
  --input /app/examples/smoke_input.csv `
  --output /work/smoke_prepped_SMILES.csv `
  --discarded-output /work/smoke_discarded_molecules.csv `
  --canonical-output /work/smoke_SMILES_canonical.csv `
  --dimorph-output /work/smoke_dimorph_canonical.csv `
  --canonical-log /work/smoke_out_canonicalizer.txt `
  --dimorph-log /work/smoke_out_dimorphite.txt `
  --workers 1
```

Bash:

```bash
mkdir -p outputs
docker run --rm \
  --gpus all \
  -v "$PWD/outputs:/work" \
  ilp:latest \
  --input /app/examples/smoke_input.csv \
  --output /work/smoke_prepped_SMILES.csv \
  --discarded-output /work/smoke_discarded_molecules.csv \
  --canonical-output /work/smoke_SMILES_canonical.csv \
  --dimorph-output /work/smoke_dimorph_canonical.csv \
  --canonical-log /work/smoke_out_canonicalizer.txt \
  --dimorph-log /work/smoke_out_dimorphite.txt \
  --workers 1
```

Expected result: `outputs/smoke_prepped_SMILES.csv` exists and contains a header plus one prepared SMILES row per smoke-test ID. `outputs/smoke_discarded_molecules.csv` should contain only the header for the bundled smoke input.

The bundled full `input.csv` took about 19 minutes on CPU on the test machine. With `--gpus all`, TensorFlow should print the detected GPU before model inference starts. If TensorFlow cannot use the GPU, it prints that no GPU was detected and runs on CPU. On very new NVIDIA cards, TensorFlow 2.7 may spend several minutes JIT-compiling CUDA kernels; the image stores CUDA cache data in `/work/.cuda_cache` so repeated runs with the same mounted work directory can reuse it when supported by the driver.

## Discard Report

`discarded_molecules.csv` contains molecules that do not produce a final output row. The columns are:

| Column | Meaning |
| --- | --- |
| `stage` | `canonicalize`, `protonate`, or `prepare`. |
| `reason` | Why the molecule was discarded. |
| `target` | The input ID. |
| `input_smiles` | Original input SMILES when the discard happened during canonicalization. |
| `canonical_smiles` | Canonical SMILES when the discard happened after canonicalization. |
| `protonation_group` | Internal group ID used for protonation-state selection. |
| `protonation_smiles_example` | Example protonation-state SMILES for preparation-stage discards. |
| `details` | Counts or error details for diagnosis. |

## Run on Your Input

Your input must be a CSV with a header and at least two columns:

```csv
smiles,id
CCO,mol1
CCN,mol2
```

Run with the current folder mounted as `/work`:

```powershell
docker run --rm `
  --gpus all `
  -v "${PWD}:/work" `
  ilp:latest `
  --input /work/input.csv `
  --output /work/prepped_SMILES.csv `
  --discarded-output /work/discarded_molecules.csv `
  --canonical-output /work/SMILES_canonical.csv `
  --dimorph-output /work/dimorph_canonical.csv
```

## CLI Options

Common options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--input` | `input.csv` | Input CSV with `smiles,id` header. |
| `--output` | `prepped_SMILES.csv` | Final output CSV. |
| `--discarded-output` | `discarded_molecules.csv` | Molecules that fail at canonicalization, protonation, or preparation. |
| `--canonical-output` | `SMILES_canonical.csv` | Intermediate canonical SMILES file. |
| `--dimorph-output` | `dimorph_canonical.csv` | Intermediate protonation-state file. |
| `--workers` | up to `4` | Worker processes for RDKit and Dimorphite stages. |
| `--prediction-batch-size` | `10000` | Max scored rows per inference batch, without splitting molecule groups. |
| `--keras-batch-size` | `512` | TensorFlow/Keras inference batch size. |
| `--min-ph` / `--max-ph` | `5.0` / `9.0` | Protonation pH window. |
| `--max-variants` | `265` | Maximum protonation variants per canonical molecule. |
| `--tie-policy` | `first` | Keep one best protonation state per molecule. `all` is available only for exploratory tie inspection. |
| `--require-gpu` | off | Fail unless TensorFlow sees a GPU. |
| `--allow-cpu` | off | Force CPU mode even if `ILP_REQUIRE_GPU=1` is set externally. |
| `--start-at` | `canonicalize` | Resume at `canonicalize`, `protonate`, or `prepare`. |
| `--stop-after` | `prepare` | Stop after `canonicalize`, `protonate`, or `prepare`. |

Example: resume from protonation after canonicalization has already finished:

```bash
docker run --rm --gpus all -v "$PWD:/work" ilp:latest \
  --start-at protonate \
  --canonical-output /work/SMILES_canonical.csv \
  --dimorph-output /work/dimorph_canonical.csv \
  --output /work/prepped_SMILES.csv
```

## Local Development

The recommended development/runtime path is Docker because the saved H5 models were built with Keras/TensorFlow 2.7-era serialization.

If you create a local environment, use Python 3.8 and TensorFlow 2.7.1, then install:

```bash
python -m pip install -r requirements.txt
```

Run the unit test inside the Docker image:

```bash
docker run --rm --entrypoint python ilp:latest -m unittest discover -s /app/tests
```

## Git Push Checklist

This project contains large model files. The included `.gitattributes` marks `.h5` and `.pkl` files for Git LFS.

Before the first push:

```bash
git lfs install
git add .gitattributes
git add Dockerfile requirements.txt README.md iLP_run.py examples tests
git add iLP/*.h5 MolAI/*.h5 MolAI/*.pkl
git commit -m "Prepare iLP Docker runtime"
```

Generated pipeline outputs are ignored by `.gitignore`.

## Runtime Notes

- The Docker image uses `tensorflow/tensorflow:2.7.1-gpu`.
- Docker should be run with `--gpus all` when GPU acceleration is desired. Without a visible compatible GPU, the tool falls back to CPU unless `--require-gpu` is set.
- The modern PyPI package `dimorphite_dl` now targets newer Python versions, so this image installs `dimorphite-ojmb==1.2.5.post1`, which provides the compatible `dimorphite_dl.Protonator` API on Python 3.8.
- `--tie-policy first` is the default and produces one output row per prepared molecule group. `--tie-policy all` preserves exact max-score ties only when you explicitly request that behavior.
