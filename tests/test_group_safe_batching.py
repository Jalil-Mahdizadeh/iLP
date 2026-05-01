import unittest
import csv
import re
import tempfile
from pathlib import Path

from ilp.pipeline import (
    DEFAULT_ALLOWED_CHARACTERS,
    iter_filtered_protonation_groups,
    iter_group_safe_batches,
    score_batch,
)


class DummyBundle:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.offset = 0

    def __call__(self, x_batch):
        import torch

        end = self.offset + len(x_batch)
        values = self.probabilities[self.offset:end]
        self.offset = end
        return torch.tensor(values, dtype=torch.float32)


class GroupSafeBatchingTests(unittest.TestCase):
    def test_batches_never_split_groups(self):
        groups = [
            [("smiles-a1", "id-a", "group-a"), ("smiles-a2", "id-a", "group-a")],
            [
                ("smiles-b1", "id-b", "group-b"),
                ("smiles-b2", "id-b", "group-b"),
                ("smiles-b3", "id-b", "group-b"),
            ],
            [("smiles-c1", "id-c", "group-c")],
        ]

        batches = list(iter_group_safe_batches(groups, batch_size=4))

        self.assertEqual(len(batches), 2)
        self.assertEqual([row[2] for row in batches[0]], ["group-a", "group-a"])
        self.assertEqual(
            [row[2] for row in batches[1]],
            ["group-b", "group-b", "group-b", "group-c"],
        )

    def test_filtered_groups_include_empty_groups_for_discard_reporting(self):
        escaped = "".join(re.escape(char) for char in DEFAULT_ALLOWED_CHARACTERS)
        allowed_regex = re.compile(f"[^{escaped}]")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dimorph.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["CCO", "id-ok", "0"])
                writer.writerow(["CCZ", "id-bad", "1"])

            groups = list(
                iter_filtered_protonation_groups(
                    path,
                    allowed_regex=allowed_regex,
                    min_length=2,
                    max_length=110,
                )
            )

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0].rows), 1)
        self.assertEqual(groups[1].target, "id-bad")
        self.assertEqual(groups[1].rows, [])
        self.assertEqual(groups[1].invalid_character_rows, 1)

    def test_score_batch_selects_one_state_per_group(self):
        rows, group_count = score_batch(
            [
                ("CCO", "mol-a", "group-a"),
                ("CCN", "mol-a", "group-a"),
                ("CO", "mol-b", "group-b"),
                ("CN", "mol-b", "group-b"),
            ],
            bundle=DummyBundle([0.2, 0.8, 0.7, 0.700001]),
            device="cpu",
            char_to_int={"!": 0, "$": 1, "%": 2, "C": 3, "O": 4, "N": 5},
            max_smi_len=8,
            inference_batch_size=2,
            tie_policy="first",
            selection_atol=2e-5,
        )

        self.assertEqual(group_count, 2)
        self.assertEqual(rows, [("CCN", "mol-a"), ("CO", "mol-b")])


if __name__ == "__main__":
    unittest.main()
