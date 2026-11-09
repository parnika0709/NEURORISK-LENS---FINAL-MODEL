"""Patient-level splits.

CRITICAL: split by ``patient_id``, not by slice or case. A patient may appear
in multiple slices but must NOT appear in both train and val/test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from braintumor.data.brats_loader import BratsCase, discover_cases


@dataclass
class PatientSplit:
    train: list[str]
    val: list[str]
    test: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PatientSplit":
        data = json.loads(path.read_text())
        return cls(train=data["train"], val=data["val"], test=data["test"])


def patient_level_split(
    cases: list[BratsCase],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> PatientSplit:
    """Deterministic patient-level split.

    No patient appears in more than one split. Ratios sum to 1.0.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    # Group by patient_id to enforce patient-level separation.
    by_patient: dict[str, list[BratsCase]] = {}
    for case in cases:
        by_patient.setdefault(case.patient_id, []).append(case)

    patients = sorted(by_patient.keys())
    import random

    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Test gets the remainder to ensure all patients are assigned.
    n_test = n - n_train - n_val

    return PatientSplit(
        train=patients[:n_train],
        val=patients[n_train : n_train + n_val],
        test=patients[n_train + n_val : n_train + n_val + n_test],
    )


def cases_for_patients(
    cases: list[BratsCase],
    patients: list[str],
) -> list[BratsCase]:
    """Filter a case list to a given set of patient IDs."""
    patient_set = set(patients)
    return [c for c in cases if c.patient_id in patient_set]


def assert_no_patient_overlap(*splits: PatientSplit) -> None:
    """Raise if a patient appears in more than one split."""
    seen: dict[str, str] = {}
    for split in splits:
        for split_name, patients in [("train", split.train), ("val", split.val), ("test", split.test)]:
            for pid in patients:
                if pid in seen:
                    raise AssertionError(
                        f"Patient {pid} appears in both '{seen[pid]}' and '{split_name}'."
                    )
                seen[pid] = split_name


def discover_and_split(
    root: Path,
    split_file: Path | None = None,
    seed: int = 42,
) -> tuple[list[BratsCase], list[BratsCase], list[BratsCase]]:
    """Discover cases under ``root`` and return (train, val, test) case lists.

    If ``split_file`` exists, it is loaded (reproducible). Otherwise a new
    split is generated and saved.
    """
    cases = discover_cases(root)
    if not cases:
        raise FileNotFoundError(f"No cases discovered under {root}")

    if split_file is not None and split_file.exists():
        split = PatientSplit.load(split_file)
    else:
        split = patient_level_split(cases, seed=seed)
        if split_file is not None:
            split.save(split_file)

    train = cases_for_patients(cases, split.train)
    val = cases_for_patients(cases, split.val)
    test = cases_for_patients(cases, split.test)

    assert_no_patient_overlap(split)
    return train, val, test
