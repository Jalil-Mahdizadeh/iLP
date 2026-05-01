# iLP Ligand Preparation Tool

iLP prepares ligand SMILES for downstream modeling. It canonicalizes and desalts input molecules, enumerates likely protonation states, scores the generated states with the bundled iLP neural model, and writes exactly one best protonation state for each input molecule that can be prepared successfully.

The Docker image uses PyTorch for model inference. CUDA is used automatically when Docker exposes a compatible NVIDIA GPU; otherwise the same command falls back to CPU.

## Citation

If you use this repository, please cite the related MolAI paper:

Mahdizadeh, S. J.; Eriksson, L. A. MolAI: A Deep Learning Framework for Data-Driven Molecular Descriptor Generation and Advanced Drug Discovery Applications. *Journal of Chemical Information and Modeling* **2025**, *65* (19), 9892-9909. https://doi.org/10.1021/acs.jcim.5c00491

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/ilp/` | Ligand preparation package and CLI pipeline. |
| `models/ilp/` | Bundled runtime model checkpoint and tokenizer. |
| `examples/` | Small smoke-test and discard-report inputs. |
| `input.csv` | Included benchmark workload used for the runtime numbers below. |
| `tests/` | Unit tests for batching, one-best selection, and discard tracking. |
| `Dockerfile` | Reproducible CUDA-capable runtime image. |
| `CITATION.cff` | Citation metadata for the related paper. |

## Build

```bash
docker build -t ilp:latest .
```

## Input Format

Use a CSV file with a header and at least two columns. The first column is treated as SMILES and the second as the molecule ID:

```csv
smiles,id
CCO,mol1
CCN,mol2
```

## Run

GPU-capable run:

```bash
docker run --rm \
  --gpus all \
  -v "$PWD:/work" \
  ilp:latest \
  --input /work/input.csv \
  --output /work/prepped_SMILES.csv \
  --discarded-output /work/discarded_molecules.csv
```

CPU-only run:

```bash
docker run --rm \
  -v "$PWD:/work" \
  ilp:latest \
  --device cpu \
  --input /work/input.csv \
  --output /work/prepped_SMILES.csv \
  --discarded-output /work/discarded_molecules.csv
```

Useful optional outputs:

```bash
  --canonical-output /work/SMILES_canonical.csv \
  --dimorph-output /work/dimorph_canonical.csv \
  --canonical-log /work/out_canonicalizer.txt \
  --dimorph-log /work/out_dimorphite.txt
```

## Outputs

| File | Meaning |
| --- | --- |
| `prepped_SMILES.csv` | Final ligand-preparation output with one selected protonation-state SMILES per input molecule. |
| `discarded_molecules.csv` | Molecules that failed canonicalization, protonation, or preparation. |
| `SMILES_canonical.csv` | Canonical/desalted intermediate file. |
| `dimorph_canonical.csv` | Protonation-state intermediate file. |
| `out_canonicalizer.txt` | Canonicalization summary. |
| `out_dimorphite.txt` | Protonation summary. |

`discarded_molecules.csv` includes the failed stage, reason, target ID, source SMILES when available, and diagnostic details.

## GPU Behavior

The runtime image is based on `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`. It is intended for Blackwell-generation GPUs and many earlier NVIDIA GPUs supported by the PyTorch CUDA 12.8 runtime.

Device controls:

| Option | Behavior |
| --- | --- |
| `--device auto` | Default. Use CUDA if available; otherwise use CPU. |
| `--device cuda` | Require CUDA for inference. |
| `--device cpu` | Force CPU inference. |
| `--require-gpu` | Fail the run if CUDA is not visible. |
| `--allow-cpu` | Override `--require-gpu` or `ILP_REQUIRE_GPU=1`. |

## Benchmarks

Benchmarks on the included `input.csv` workload, measured on the same machine:

| Metric | Value |
| --- | --- |
| Input molecules | `9,565` |
| Protonation rows generated | `113,075` |
| Filtered protonation rows scored | `112,926` |
| Final output rows | `9,565` |
| Discarded molecules | `0` |

| Mode | Hardware / command mode | Total runtime |
| --- | --- | --- |
| GPU | NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU, `--gpus all` | `59.5 s` |
| CPU | Docker CPU run with `--device cpu` | about `25 min` |

The CPU and GPU full-output files were byte-identical after deterministic near-tie handling.

## CLI Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input` | `input.csv` | Input CSV path. |
| `--output` | `prepped_SMILES.csv` | Final output CSV. |
| `--discarded-output` | `discarded_molecules.csv` | Discard report CSV. |
| `--model-dir` | bundled model | Runtime model directory. |
| `--workers` | up to `4` | Worker processes for RDKit and Dimorphite stages. |
| `--prediction-batch-size` | `10000` | Maximum scored rows per group-safe scoring batch. |
| `--inference-batch-size` | `512` | Tensor batch size inside each scoring batch. |
| `--tie-policy` | `first` | Keep one best protonation state per molecule. |
| `--selection-atol` | `2e-5` | Probability tolerance for deterministic CPU/GPU near-tie handling. |
| `--start-at` | `canonicalize` | Resume at `canonicalize`, `protonate`, or `prepare`. |
| `--stop-after` | `prepare` | Stop after `canonicalize`, `protonate`, or `prepare`. |

## Development Checks

```bash
docker run --rm --entrypoint python ilp:latest -m unittest discover -s /app/tests
```

The bundled `models/ilp/ilp_model.pt` file is tracked with Git LFS.

Generated pipeline outputs, CUDA cache files, and benchmark outputs are ignored by Git.
