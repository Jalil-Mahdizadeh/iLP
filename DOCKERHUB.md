# iLP Ligand Preparation Tool

`iLP` is a Dockerized ligand-preparation workflow for SMILES input files. It canonicalizes and desalts molecules, enumerates likely protonation states, scores the generated states with the bundled iLP neural model, and writes one selected protonation-state SMILES for each input molecule that can be prepared successfully.

The image is designed for reproducible command-line use on both CPU-only systems and NVIDIA GPU systems. When a compatible GPU is exposed to Docker, iLP uses CUDA automatically. If no compatible GPU is available, it runs on CPU.

## What This Image Does

The container runs the full iLP preparation pipeline:

1. Reads an input CSV containing SMILES and molecule IDs.
2. Canonicalizes and desalts molecules with RDKit.
3. Enumerates protonation states with Dimorphite.
4. Scores protonation-state candidates with the bundled iLP model.
5. Selects exactly one best protonation state per input molecule.
6. Writes a final prepared-SMILES output file.
7. Writes a discard report for molecules that could not be prepared.

## Image Highlights

- CPU and GPU support in one image.
- CUDA is selected automatically when available.
- Compatible with modern NVIDIA GPUs, including Blackwell-generation hardware, through the PyTorch CUDA 12.8 runtime.
- Produces one best protonation state per input molecule by default.
- Writes a `discarded_molecules.csv` report for failed molecules.
- Includes optional intermediate files for debugging and reproducibility.
- Runs fully from Docker without requiring a local Python environment.

## Image Name

```bash
docker pull 951753jalil/ilp:latest
```

Example after pulling:

```bash
docker run --rm 951753jalil/ilp:latest --help
```

## Input Format

The input must be a CSV file with a header and at least two columns. The first column is treated as the SMILES string and the second column is treated as the molecule ID.

```csv
smiles,id
CCO,mol1
CCN,mol2
```

Additional columns may be present, but the pipeline uses the first two columns.

## Quick Start

Run from a directory containing `input.csv`.

### GPU Run

```bash
docker run --rm \
  --gpus all \
  -v "$PWD:/work" \
  951753jalil/ilp:latest \
  --input /work/input.csv \
  --output /work/prepped_SMILES.csv \
  --discarded-output /work/discarded_molecules.csv
```

### CPU-Only Run

```bash
docker run --rm \
  -v "$PWD:/work" \
  951753jalil/ilp:latest \
  --device cpu \
  --input /work/input.csv \
  --output /work/prepped_SMILES.csv \
  --discarded-output /work/discarded_molecules.csv
```

## Recommended Full Command

This command writes the final output, discard report, intermediate files, and logs:

```bash
docker run --rm \
  --gpus all \
  -v "$PWD:/work" \
  951753jalil/ilp:latest \
  --input /work/input.csv \
  --output /work/prepped_SMILES.csv \
  --discarded-output /work/discarded_molecules.csv \
  --canonical-output /work/SMILES_canonical.csv \
  --dimorph-output /work/dimorph_canonical.csv \
  --canonical-log /work/out_canonicalizer.txt \
  --dimorph-log /work/out_dimorphite.txt
```

For CPU-only execution, remove `--gpus all` and add `--device cpu`.

## Outputs

| File | Description |
| --- | --- |
| `prepped_SMILES.csv` | Final prepared ligand file with one selected protonation-state SMILES per input molecule. |
| `discarded_molecules.csv` | Molecules that failed canonicalization, protonation, or final preparation. |
| `SMILES_canonical.csv` | Canonicalized and desalted intermediate SMILES file. |
| `dimorph_canonical.csv` | Enumerated protonation-state intermediate file. |
| `out_canonicalizer.txt` | Canonicalization summary log. |
| `out_dimorphite.txt` | Protonation summary log. |

The final output has this structure:

```csv
prepped_SMILES,target
CCO,mol1
CC[NH3+],mol2
```

The discard report includes diagnostic columns:

| Column | Description |
| --- | --- |
| `stage` | Stage where the molecule failed: `canonicalize`, `protonate`, or `prepare`. |
| `reason` | Failure reason. |
| `target` | Molecule ID from the input file. |
| `input_smiles` | Original input SMILES when available. |
| `canonical_smiles` | Canonical SMILES when available. |
| `protonation_group` | Internal protonation group ID. |
| `protonation_smiles_example` | Example protonation-state SMILES when available. |
| `details` | Additional diagnostic counts or error details. |

## GPU and CPU Behavior

The default device mode is:

```bash
--device auto
```

In this mode:

- iLP uses CUDA when Docker exposes a compatible NVIDIA GPU.
- iLP falls back to CPU when CUDA is not available.

Device-control options:

| Option | Behavior |
| --- | --- |
| `--device auto` | Default. Use CUDA if available; otherwise use CPU. |
| `--device cuda` | Require CUDA for inference. |
| `--device cpu` | Force CPU inference. |
| `--require-gpu` | Fail if CUDA is not visible. |
| `--allow-cpu` | Override `--require-gpu` or `ILP_REQUIRE_GPU=1`. |

To verify GPU access:

```bash
docker run --rm --gpus all 951753jalil/ilp:latest --help
```

During a real run, the container prints the detected CUDA device name when GPU inference is active.

## Performance

Benchmarks on the included `input.csv` workload:

| Metric | Value |
| --- | --- |
| Input molecules | `9,565` |
| Protonation rows generated | `113,075` |
| Filtered protonation rows scored | `112,926` |
| Final output rows | `9,565` |
| Discarded molecules | `0` |

Measured runtimes on the same machine:

| Mode | Runtime |
| --- | --- |
| GPU | `59.5 s` |
| CPU | about `25 min` |

Actual runtime depends on CPU, GPU, storage speed, Docker configuration, and input molecule complexity.

## Common CLI Options

| Option | Default | Description |
| --- | --- | --- |
| `--input` | `input.csv` | Input CSV path. |
| `--output` | `prepped_SMILES.csv` | Final prepared-SMILES output file. |
| `--discarded-output` | `discarded_molecules.csv` | Discard report path. |
| `--canonical-output` | `SMILES_canonical.csv` | Canonicalized intermediate file. |
| `--dimorph-output` | `dimorph_canonical.csv` | Protonation-state intermediate file. |
| `--canonical-log` | `out_canonicalizer.txt` | Canonicalization log path. |
| `--dimorph-log` | `out_dimorphite.txt` | Protonation log path. |
| `--workers` | up to `4` | Worker processes for RDKit and Dimorphite stages. |
| `--prediction-batch-size` | `10000` | Maximum scored rows per group-safe scoring batch. |
| `--inference-batch-size` | `512` | Neural-model inference batch size. |
| `--tie-policy` | `first` | Select one best state per molecule. |
| `--selection-atol` | `2e-5` | Near-tie tolerance for deterministic CPU/GPU selection. |
| `--start-at` | `canonicalize` | Resume from `canonicalize`, `protonate`, or `prepare`. |
| `--stop-after` | `prepare` | Stop after `canonicalize`, `protonate`, or `prepare`. |

## Resume From Intermediate Files

You can resume from a later stage if intermediate files already exist.

Resume from protonation:

```bash
docker run --rm \
  -v "$PWD:/work" \
  951753jalil/ilp:latest \
  --start-at protonate \
  --canonical-output /work/SMILES_canonical.csv \
  --dimorph-output /work/dimorph_canonical.csv \
  --output /work/prepped_SMILES.csv \
  --discarded-output /work/discarded_molecules.csv
```

Run only final preparation from an existing `dimorph_canonical.csv`:

```bash
docker run --rm \
  --gpus all \
  -v "$PWD:/work" \
  951753jalil/ilp:latest \
  --start-at prepare \
  --dimorph-output /work/dimorph_canonical.csv \
  --output /work/prepped_SMILES.csv \
  --discarded-output /work/discarded_molecules.csv
```

## Requirements

For CPU execution:

- Docker

For GPU execution:

- Docker
- NVIDIA GPU
- NVIDIA driver compatible with CUDA 12.8 runtime images
- NVIDIA Container Toolkit
- `docker run --gpus all`

## Citation

If you use this image or the repository in scientific work, please cite:

Mahdizadeh, S. J.; Eriksson, L. A. MolAI: A Deep Learning Framework for Data-Driven Molecular Descriptor Generation and Advanced Drug Discovery Applications. *Journal of Chemical Information and Modeling* **2025**, *65* (19), 9892-9909. https://doi.org/10.1021/acs.jcim.5c00491

## Suggested Docker Hub Short Description

```text
Dockerized iLP ligand preparation tool for SMILES canonicalization, protonation-state selection, and CPU/GPU-ready neural scoring.
```

## Suggested Tags

```text
ligand-preparation, smiles, cheminformatics, rdkit, dimorphite, molecular-modeling, gpu, cuda, pytorch
```
