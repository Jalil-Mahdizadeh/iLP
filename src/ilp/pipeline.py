"""Run the iLP SMILES preparation pipeline.

The pipeline canonicalizes/desalts input SMILES, enumerates protonation states,
and uses the MolAI encoder plus the iLP ensemble to choose protonation-state
SMILES for downstream modeling work.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from tqdm import tqdm

from .neural import load_bundle


SCRIPT_DIR = Path(__file__).resolve().parents[2]
DISCARD_COLUMNS = [
    "stage",
    "reason",
    "target",
    "input_smiles",
    "canonical_smiles",
    "protonation_group",
    "protonation_smiles_example",
    "details",
]
DEFAULT_ALLOWED_CHARACTERS = {
    "c",
    "3",
    "5",
    "=",
    "O",
    "+",
    "8",
    "[",
    "s",
    "]",
    "C",
    "n",
    "p",
    "H",
    "%",
    "-",
    "(",
    "I",
    "1",
    "4",
    "Y",
    "X",
    "6",
    "!",
    "S",
    ")",
    "#",
    "o",
    "2",
    "P",
    "N",
    "7",
    "F",
}

_CANONICAL_REMOVER = None
_DIMORPHITE = None


@dataclass
class PipelineSummary:
    input_rows: int = 0
    canonical_rows: int = 0
    canonical_discards: int = 0
    protonated_rows: int = 0
    protonated_groups: int = 0
    protonation_discards: int = 0
    filtered_protonation_rows: int = 0
    prediction_batches: int = 0
    output_rows: int = 0
    output_groups: int = 0
    prepare_discards: int = 0


@dataclass
class ProtonationGroup:
    group_id: str
    target: str
    protonation_smiles_example: str
    rows: List[Tuple[str, str, str]]
    total_rows: int
    invalid_character_rows: int
    too_short_rows: int
    too_long_rows: int


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_discard(
    writer: csv.writer | None,
    stage: str,
    reason: str,
    target: str = "",
    input_smiles: str = "",
    canonical_smiles: str = "",
    protonation_group: str = "",
    protonation_smiles_example: str = "",
    details: str = "",
) -> None:
    if writer is None:
        return
    writer.writerow(
        [
            stage,
            reason,
            target,
            input_smiles,
            canonical_smiles,
            protonation_group,
            protonation_smiles_example,
            details,
        ]
    )


def _init_canonical_worker() -> None:
    global _CANONICAL_REMOVER
    _CANONICAL_REMOVER = SaltRemover.SaltRemover()


def generate_canonical_smiles(smi: str) -> Tuple[str | None, str]:
    """Return a canonical, desalted SMILES plus an empty/error reason."""
    global _CANONICAL_REMOVER
    if _CANONICAL_REMOVER is None:
        _CANONICAL_REMOVER = SaltRemover.SaltRemover()

    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None, "rdkit_parse_failed"

    desalted = _CANONICAL_REMOVER.StripMol(mol, dontRemoveEverything=True)
    if desalted.GetNumAtoms() == 0:
        return None, "desalt_removed_all_atoms"
    return Chem.MolToSmiles(desalted, canonical=True, isomericSmiles=False), ""


def _pool_map(
    func,
    values: Sequence,
    workers: int,
    pool_chunksize: int,
    initializer=None,
    initargs: tuple = (),
) -> Iterable:
    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        return map(func, values)

    pool = multiprocessing.Pool(
        processes=workers,
        initializer=initializer,
        initargs=initargs,
    )
    return _closing_imap(pool, func, values, pool_chunksize)


def _closing_imap(pool, func, values: Sequence, pool_chunksize: int) -> Iterator:
    try:
        yield from pool.imap(func, values, chunksize=pool_chunksize)
    finally:
        pool.close()
        pool.join()


def canonicalize_csv(
    input_file: Path,
    output_file: Path,
    chunksize: int,
    workers: int,
    pool_chunksize: int,
    discarded_writer: csv.writer | None = None,
) -> Tuple[int, int, int]:
    _ensure_parent(output_file)
    input_rows = 0
    canonical_rows = 0
    discarded_rows = 0

    with output_file.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.writer(out_handle)
        for chunk in tqdm(
            pd.read_csv(input_file, chunksize=chunksize),
            desc="Canonicalizing",
            unit="chunk",
        ):
            if chunk.shape[1] < 2:
                raise ValueError(f"{input_file} must contain at least two columns: smiles,id")

            smiles = chunk.iloc[:, 0].astype(str).tolist()
            targets = chunk.iloc[:, 1].astype(str).tolist()
            input_rows += len(smiles)

            mapped = _pool_map(
                generate_canonical_smiles,
                smiles,
                workers=workers,
                pool_chunksize=pool_chunksize,
                initializer=_init_canonical_worker,
            )
            for input_smiles, target, result in zip(smiles, targets, mapped):
                canonical_smiles, reason = result
                if canonical_smiles is not None:
                    writer.writerow([canonical_smiles, target])
                    canonical_rows += 1
                else:
                    _write_discard(
                        discarded_writer,
                        stage="canonicalize",
                        reason=reason,
                        target=target,
                        input_smiles=input_smiles,
                    )
                    discarded_rows += 1

    return input_rows, canonical_rows, discarded_rows


def _build_dimorphite(
    min_ph: float,
    max_ph: float,
    max_variants: int,
    label_states: bool,
    pka_precision: float,
):
    try:
        from dimorphite_dl import DimorphiteDL

        try:
            return DimorphiteDL(
                min_ph=min_ph,
                max_ph=max_ph,
                max_variants=max_variants,
                label_states=label_states,
                pka_precision=pka_precision,
            )
        except TypeError:
            return DimorphiteDL(
                ph_min=min_ph,
                ph_max=max_ph,
                max_variants=max_variants,
                label_states=label_states,
                precision=pka_precision,
            )
    except ImportError:
        from dimorphite_dl import Protonator

        return Protonator(
            min_ph=min_ph,
            max_ph=max_ph,
            pka_precision=pka_precision,
            max_variants=max_variants,
            label_states=label_states,
            silent=True,
        )


def _init_dimorphite_worker(config: dict) -> None:
    global _DIMORPHITE
    _DIMORPHITE = _build_dimorphite(**config)


def generate_protonations(smile: str) -> Tuple[List[str], str]:
    global _DIMORPHITE
    if _DIMORPHITE is None:
        raise RuntimeError("Dimorphite worker was not initialized")

    try:
        states = _DIMORPHITE.protonate(str(smile))
    except Exception as exc:  # noqa: BLE001 - keep per-molecule failures reportable.
        return [], f"dimorphite_error:{type(exc).__name__}:{exc}"

    normalized_states = []
    for state in states:
        if isinstance(state, (tuple, list)):
            normalized_states.append(str(state[0]))
        else:
            normalized_states.append(str(state))
    if not normalized_states:
        return [], "no_protonation_states"
    return normalized_states, ""


def protonate_csv(
    input_file: Path,
    output_file: Path,
    chunksize: int,
    workers: int,
    pool_chunksize: int,
    min_ph: float,
    max_ph: float,
    max_variants: int,
    label_states: bool,
    pka_precision: float,
    discarded_writer: csv.writer | None = None,
) -> Tuple[int, int, int]:
    _ensure_parent(output_file)
    molecule_count = 0
    protonation_rows = 0
    discarded_rows = 0
    config = {
        "min_ph": min_ph,
        "max_ph": max_ph,
        "max_variants": max_variants,
        "label_states": label_states,
        "pka_precision": pka_precision,
    }

    with output_file.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.writer(out_handle)
        for chunk in tqdm(
            pd.read_csv(input_file, chunksize=chunksize, header=None),
            desc="Protonating",
            unit="chunk",
        ):
            if chunk.shape[1] < 2:
                raise ValueError(f"{input_file} must contain at least two columns")

            smiles = chunk.iloc[:, 0].astype(str).tolist()
            targets = chunk.iloc[:, 1].astype(str).tolist()
            mapped = _pool_map(
                generate_protonations,
                smiles,
                workers=workers,
                pool_chunksize=pool_chunksize,
                initializer=_init_dimorphite_worker,
                initargs=(config,),
            )

            for canonical_smiles, target, result in zip(smiles, targets, mapped):
                protonation_states, reason = result
                if not protonation_states:
                    _write_discard(
                        discarded_writer,
                        stage="protonate",
                        reason=reason,
                        target=target,
                        canonical_smiles=canonical_smiles,
                        protonation_group=str(molecule_count),
                    )
                    discarded_rows += 1
                for protonation_state in protonation_states:
                    writer.writerow([protonation_state, target, molecule_count])
                    protonation_rows += 1
                molecule_count += 1

    return molecule_count, protonation_rows, discarded_rows


def vectorize_smiles(
    smiles: Sequence[str],
    char_to_int: dict,
    max_smi_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    one_hot = np.zeros((len(smiles), max_smi_len, len(char_to_int)), dtype=np.float32)
    for i, smile in enumerate(smiles):
        one_hot[i, 0, char_to_int["!"]] = 1
        for j, char in enumerate(smile):
            if char in char_to_int:
                one_hot[i, j + 1, char_to_int[char]] = 1
        one_hot[i, len(smile) + 1, char_to_int["$"]] = 1
        one_hot[i, len(smile) + 2 :, char_to_int["%"]] = 1
    return one_hot[:, 1:, :], one_hot[:, :-1, :]


def _normalize_halogen_tokens(smiles: str) -> str:
    return str(smiles).replace("Cl", "X").replace("Br", "Y")


def _restore_halogen_tokens(smiles: str) -> str:
    return str(smiles).replace("X", "Cl").replace("Y", "Br")


def iter_filtered_protonation_groups(
    input_file: Path,
    allowed_regex: re.Pattern,
    min_length: int,
    max_length: int,
) -> Iterator[ProtonationGroup]:
    current_group = None
    current_target = ""
    current_example = ""
    current_rows: List[Tuple[str, str, str]] = []
    current_total_rows = 0
    current_invalid_character_rows = 0
    current_too_short_rows = 0
    current_too_long_rows = 0

    def make_group() -> ProtonationGroup:
        return ProtonationGroup(
            group_id=str(current_group),
            target=current_target,
            protonation_smiles_example=current_example,
            rows=current_rows,
            total_rows=current_total_rows,
            invalid_character_rows=current_invalid_character_rows,
            too_short_rows=current_too_short_rows,
            too_long_rows=current_too_long_rows,
        )

    with input_file.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 3:
                raise ValueError(f"{input_file}:{row_number} must contain SMILES,target,group")

            raw_smiles, target, group = row[0], row[1], row[2]
            if current_group is None:
                current_group = group
                current_target = str(target)
                current_example = str(raw_smiles)
            elif group != current_group:
                yield make_group()
                current_group = group
                current_target = str(target)
                current_example = str(raw_smiles)
                current_rows = []
                current_total_rows = 0
                current_invalid_character_rows = 0
                current_too_short_rows = 0
                current_too_long_rows = 0

            smiles = _normalize_halogen_tokens(raw_smiles)
            current_total_rows += 1
            if allowed_regex.search(smiles):
                current_invalid_character_rows += 1
                continue
            if len(smiles) <= min_length:
                current_too_short_rows += 1
                continue
            if len(smiles) >= max_length:
                current_too_long_rows += 1
                continue

            current_rows.append((smiles, str(target), str(group)))

    if current_group is not None:
        yield make_group()


def iter_group_safe_batches(
    groups: Iterable[List[Tuple[str, str, str]]],
    batch_size: int,
) -> Iterator[List[Tuple[str, str, str]]]:
    batch: List[Tuple[str, str, str]] = []

    for group_rows in groups:
        if batch and len(batch) + len(group_rows) > batch_size:
            yield batch
            batch = []
        batch.extend(group_rows)

    if batch:
        yield batch


def load_models(model_dir: Path, requested_device: str, require_gpu: bool):
    import torch

    if require_gpu:
        requested_device = "cuda"
    bundle, char_to_int, int_to_char, device, metadata = load_bundle(
        model_dir,
        requested_device=requested_device,
    )
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        print(
            f"PyTorch CUDA device detected: {name} "
            f"(compute capability {capability[0]}.{capability[1]}, CUDA {torch.version.cuda})."
        )
    else:
        print("PyTorch CUDA device detected: none; running on CPU.")
    return bundle, char_to_int, int_to_char, device, metadata


def score_batch(
    batch_rows: List[Tuple[str, str, str]],
    bundle,
    device,
    char_to_int: dict,
    max_smi_len: int,
    inference_batch_size: int,
    tie_policy: str,
    selection_atol: float,
) -> Tuple[List[Tuple[str, str]], int]:
    import torch

    smiles = [row[0] for row in batch_rows]
    targets = [row[1] for row in batch_rows]
    groups = [row[2] for row in batch_rows]

    x_test, _ = vectorize_smiles(smiles, char_to_int, max_smi_len)
    probabilities = []
    with torch.inference_mode():
        for start_idx in range(0, len(x_test), inference_batch_size):
            end_idx = min(start_idx + inference_batch_size, len(x_test))
            x_batch = torch.from_numpy(x_test[start_idx:end_idx]).to(device)
            probabilities.append(bundle(x_batch).detach().cpu().numpy())
    probability = np.concatenate(probabilities, axis=0)

    result = pd.DataFrame(
        {
            "probability": probability,
            "prepped_SMILES": smiles,
            "target": targets,
            "group": groups,
        }
    )

    if tie_policy == "first":
        max_probability = result.groupby("group")["probability"].transform("max")
        near_max = result[result["probability"] >= (max_probability - selection_atol)]
        winner_idx = near_max.groupby("group", sort=False).head(1).index
        winners = result.loc[winner_idx]
    else:
        max_probability = result.groupby("group")["probability"].transform("max")
        winners = result[result["probability"] >= (max_probability - selection_atol)]

    rows = [
        (_restore_halogen_tokens(row.prepped_SMILES), row.target)
        for row in winners.itertuples(index=False)
    ]
    return rows, result["group"].nunique()


def prepare_smiles_csv(
    input_file: Path,
    output_file: Path,
    model_dir: Path,
    batch_size: int,
    inference_batch_size: int,
    max_smi_len: int,
    min_smi_len: int,
    tie_policy: str,
    selection_atol: float,
    requested_device: str,
    require_gpu: bool,
    discarded_writer: csv.writer | None = None,
) -> Tuple[int, int, int, int, int]:
    _ensure_parent(output_file)
    bundle, char_to_int, _, device, _ = load_models(
        model_dir,
        requested_device=requested_device,
        require_gpu=require_gpu,
    )
    allowed_characters = DEFAULT_ALLOWED_CHARACTERS
    missing_characters = sorted(allowed_characters - set(char_to_int.keys()))
    if missing_characters:
        raise ValueError(f"PyTorch tokenizer is missing expected tokens: {missing_characters}")

    escaped_characters = "".join(re.escape(char) for char in allowed_characters)
    allowed_regex = re.compile(f"[^{escaped_characters}]")

    raw_groups = iter_filtered_protonation_groups(
        input_file,
        allowed_regex=allowed_regex,
        min_length=min_smi_len,
        max_length=max_smi_len - 1,
    )
    prepare_discards = {"count": 0}

    def valid_groups() -> Iterator[List[Tuple[str, str, str]]]:
        for group in raw_groups:
            if group.rows:
                yield group.rows
                continue

            details = (
                f"total_protonation_rows={group.total_rows};"
                f"invalid_character_rows={group.invalid_character_rows};"
                f"too_short_rows={group.too_short_rows};"
                f"too_long_rows={group.too_long_rows}"
            )
            _write_discard(
                discarded_writer,
                stage="prepare",
                reason="all_protonation_states_filtered",
                target=group.target,
                protonation_group=group.group_id,
                protonation_smiles_example=group.protonation_smiles_example,
                details=details,
            )
            prepare_discards["count"] += 1

    groups = valid_groups()
    batches = iter_group_safe_batches(groups, batch_size=batch_size)

    filtered_rows = 0
    prediction_batches = 0
    output_rows = 0
    output_groups = 0
    with output_file.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(["prepped_SMILES", "target"])

        for batch_rows in tqdm(batches, desc="Scoring protonation groups", unit="batch"):
            filtered_rows += len(batch_rows)
            prediction_batches += 1
            rows, group_count = score_batch(
                batch_rows,
                bundle=bundle,
                device=device,
                char_to_int=char_to_int,
                max_smi_len=max_smi_len,
                inference_batch_size=inference_batch_size,
                tie_policy=tie_policy,
                selection_atol=selection_atol,
            )
            writer.writerows(rows)
            output_rows += len(rows)
            output_groups += group_count

    return (
        filtered_rows,
        prediction_batches,
        output_rows,
        output_groups,
        prepare_discards["count"],
    )


def write_log(path: Path, lines: Sequence[str]) -> None:
    _ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    require_gpu_default = os.environ.get("ILP_REQUIRE_GPU", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    parser = argparse.ArgumentParser(
        description="Run the iLP SMILES canonicalization, protonation, and preparation pipeline."
    )
    parser.add_argument("--input", default="input.csv", help="Input CSV with smiles,id header.")
    parser.add_argument("--output", default="prepped_SMILES.csv", help="Final prepared SMILES CSV.")
    parser.add_argument(
        "--discarded-output",
        default="discarded_molecules.csv",
        help="CSV report for input molecules that do not produce a prepared SMILES row.",
    )
    parser.add_argument("--canonical-output", default="SMILES_canonical.csv")
    parser.add_argument("--dimorph-output", default="dimorph_canonical.csv")
    parser.add_argument("--canonical-log", default="out_canonicalizer.txt")
    parser.add_argument("--dimorph-log", default="out_dimorphite.txt")
    parser.add_argument("--model-dir", default=str(SCRIPT_DIR / "models" / "ilp"))
    parser.add_argument("--workers", type=int, default=max(1, min(4, multiprocessing.cpu_count())))
    parser.add_argument("--pool-chunksize", type=int, default=64)
    parser.add_argument("--canonical-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--dimorph-chunk-size", type=int, default=100_000)
    parser.add_argument("--prediction-batch-size", type=int, default=10_000)
    parser.add_argument("--inference-batch-size", dest="inference_batch_size", type=int, default=512)
    parser.add_argument("--max-smi-len", type=int, default=111)
    parser.add_argument("--min-smi-len", type=int, default=2)
    parser.add_argument("--min-ph", type=float, default=5.0)
    parser.add_argument("--max-ph", type=float, default=9.0)
    parser.add_argument("--max-variants", type=int, default=265)
    parser.add_argument("--pka-precision", type=float, default=1.0)
    parser.add_argument("--label-states", action="store_true")
    parser.add_argument(
        "--tie-policy",
        choices=("all", "first"),
        default="first",
        help="How to handle equal max-probability variants within a molecule group.",
    )
    parser.add_argument(
        "--selection-atol",
        type=float,
        default=2e-5,
        help="Absolute probability tolerance for deterministic CPU/GPU near-tie handling.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        default=require_gpu_default,
        help="Fail the prepare stage unless PyTorch detects at least one CUDA GPU.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Override --require-gpu or ILP_REQUIRE_GPU=1 and allow CPU execution.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch inference device. The default uses CUDA when available and CPU otherwise.",
    )
    parser.add_argument(
        "--start-at",
        choices=("canonicalize", "protonate", "prepare"),
        default="canonicalize",
        help="Resume from an intermediate stage.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("canonicalize", "protonate", "prepare"),
        default="prepare",
        help="Stop after an intermediate stage.",
    )
    parser.add_argument(
        "--keep-rdkit-logs",
        action="store_true",
        help="Leave RDKit parser warnings enabled.",
    )
    return parser.parse_args(argv)


def _stage_index(name: str) -> int:
    return {"canonicalize": 0, "protonate": 1, "prepare": 2}[name]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if _stage_index(args.start_at) > _stage_index(args.stop_after):
        raise ValueError("--start-at cannot be later than --stop-after")
    if args.allow_cpu:
        args.require_gpu = False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if not args.keep_rdkit_logs:
        RDLogger.DisableLog("rdApp.*")

    input_file = _resolve_path(args.input)
    output_file = _resolve_path(args.output)
    discarded_output = _resolve_path(args.discarded_output)
    canonical_output = _resolve_path(args.canonical_output)
    dimorph_output = _resolve_path(args.dimorph_output)
    canonical_log = _resolve_path(args.canonical_log)
    dimorph_log = _resolve_path(args.dimorph_log)
    model_dir = _resolve_path(args.model_dir)

    summary = PipelineSummary()
    pipeline_start = time.time()

    _ensure_parent(discarded_output)
    with discarded_output.open("w", newline="", encoding="utf-8") as discard_handle:
        discarded_writer = csv.writer(discard_handle)
        discarded_writer.writerow(DISCARD_COLUMNS)

        if _stage_index(args.start_at) <= 0 <= _stage_index(args.stop_after):
            stage_start = time.time()
            input_rows, canonical_rows, canonical_discards = canonicalize_csv(
                input_file=input_file,
                output_file=canonical_output,
                chunksize=args.canonical_chunk_size,
                workers=args.workers,
                pool_chunksize=args.pool_chunksize,
                discarded_writer=discarded_writer,
            )
            summary.input_rows = input_rows
            summary.canonical_rows = canonical_rows
            summary.canonical_discards = canonical_discards
            write_log(
                canonical_log,
                [
                    f"input_rows={input_rows}",
                    f"canonical_rows={canonical_rows}",
                    f"discarded_rows={canonical_discards}",
                    f"seconds={time.time() - stage_start:.1f}",
                ],
            )
            print(
                f"Canonicalized {canonical_rows:,} of {input_rows:,} input molecules "
                f"({canonical_discards:,} discarded)."
            )

        if _stage_index(args.start_at) <= 1 <= _stage_index(args.stop_after):
            stage_start = time.time()
            protonated_groups, protonated_rows, protonation_discards = protonate_csv(
                input_file=canonical_output,
                output_file=dimorph_output,
                chunksize=args.dimorph_chunk_size,
                workers=args.workers,
                pool_chunksize=args.pool_chunksize,
                min_ph=args.min_ph,
                max_ph=args.max_ph,
                max_variants=args.max_variants,
                label_states=args.label_states,
                pka_precision=args.pka_precision,
                discarded_writer=discarded_writer,
            )
            summary.protonated_groups = protonated_groups
            summary.protonated_rows = protonated_rows
            summary.protonation_discards = protonation_discards
            write_log(
                dimorph_log,
                [
                    f"protonated_groups={protonated_groups}",
                    f"protonated_rows={protonated_rows}",
                    f"discarded_rows={protonation_discards}",
                    f"seconds={time.time() - stage_start:.1f}",
                ],
            )
            print(
                f"Generated {protonated_rows:,} protonation rows for "
                f"{protonated_groups:,} canonical molecules "
                f"({protonation_discards:,} discarded)."
            )

        if _stage_index(args.start_at) <= 2 <= _stage_index(args.stop_after):
            (
                filtered_rows,
                prediction_batches,
                output_rows,
                output_groups,
                prepare_discards,
            ) = prepare_smiles_csv(
                input_file=dimorph_output,
                output_file=output_file,
                model_dir=model_dir,
                batch_size=args.prediction_batch_size,
                inference_batch_size=args.inference_batch_size,
                max_smi_len=args.max_smi_len,
                min_smi_len=args.min_smi_len,
                tie_policy=args.tie_policy,
                selection_atol=args.selection_atol,
                requested_device=args.device,
                require_gpu=args.require_gpu,
                discarded_writer=discarded_writer,
            )
            summary.filtered_protonation_rows = filtered_rows
            summary.prediction_batches = prediction_batches
            summary.output_rows = output_rows
            summary.output_groups = output_groups
            summary.prepare_discards = prepare_discards
            print(
                f"Wrote {output_rows:,} prepared SMILES rows for {output_groups:,} groups "
                f"from {filtered_rows:,} filtered protonation rows "
                f"({prepare_discards:,} discarded)."
            )

    print(f"iLP pipeline finished in {time.time() - pipeline_start:.1f} seconds.")
    print(f"Discard report: {discarded_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
