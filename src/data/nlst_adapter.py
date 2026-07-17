"""
data/nlst_adapter.py — controlled-access NLST dataset adapter.

NLST (National Lung Screening Trial) requires an approved NCI Data Use
Agreement — see data/downloaders.py::print_nlst_instructions for the exact
authorized-access steps. This project has no such approval in this
environment and must never pretend otherwise. This module:

  - resolves a local NLST data root from an environment variable (never a
    committed path or a credential in config — see
    configs/default.yaml's data.nlst.local_root_env, default
    "NLST_DATA_ROOT"), so a user with authorized access points this at
    their own DUA-governed local copy;
  - validates that screen.csv/prsn.csv actually exist and carry the
    required columns before anything downstream trusts them;
  - reports availability as an explicit status (not found / present)
    rather than silently proceeding with partial data;
  - never uploads, logs the contents of, or otherwise exposes
    participant-level NLST rows.

If no local root is configured/found, real ingestion is unavailable and
this module says so explicitly — it does not fall back to fabricated
data. tests/test_nlst_adapter.py exercises the schema-validation logic
against small synthetic, non-real fixture rows only.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

REQUIRED_SCREEN_COLUMNS = ["pid", "CIGSMOK", "CIGAR"]
REQUIRED_PRSN_COLUMNS = ["pid", "candx"]

DEFAULT_LOCAL_ROOT_ENV = "NLST_DATA_ROOT"


class NLSTAccessError(RuntimeError):
    """Raised when NLST data is requested but not actually available —
    never silently degrades to a fixture or a fabricated value."""


@dataclass
class NLSTAvailability:
    local_root_env: str
    local_root: Optional[str]
    screen_csv_present: bool
    prsn_csv_present: bool
    screen_columns_valid: Optional[bool]
    prsn_columns_valid: Optional[bool]

    @property
    def available(self) -> bool:
        return bool(
            self.local_root and self.screen_csv_present and self.prsn_csv_present
            and self.screen_columns_valid and self.prsn_columns_valid
        )

    def to_dict(self) -> dict:
        return {
            "local_root_env": self.local_root_env, "local_root": self.local_root,
            "screen_csv_present": self.screen_csv_present, "prsn_csv_present": self.prsn_csv_present,
            "screen_columns_valid": self.screen_columns_valid, "prsn_columns_valid": self.prsn_columns_valid,
            "available": self.available,
        }


def _validate_columns(path: Path, required: List[str]) -> bool:
    try:
        header = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    return all(c in header for c in required)


def check_nlst_availability(local_root_env: str = DEFAULT_LOCAL_ROOT_ENV) -> NLSTAvailability:
    """
    Resolve NLST_DATA_ROOT (or the configured env var) and check whether
    screen.csv/prsn.csv are actually present there with the required
    columns. Never raises — callers decide whether unavailability is fatal
    for their use case (see require_nlst_available below for the fatal
    variant).
    """
    root = os.environ.get(local_root_env)
    if not root:
        return NLSTAvailability(local_root_env, None, False, False, None, None)

    root_path = Path(root)
    screen_path = root_path / "screen.csv"
    prsn_path = root_path / "prsn.csv"
    screen_present = screen_path.exists()
    prsn_present = prsn_path.exists()
    return NLSTAvailability(
        local_root_env=local_root_env,
        local_root=root,
        screen_csv_present=screen_present,
        prsn_csv_present=prsn_present,
        screen_columns_valid=_validate_columns(screen_path, REQUIRED_SCREEN_COLUMNS) if screen_present else None,
        prsn_columns_valid=_validate_columns(prsn_path, REQUIRED_PRSN_COLUMNS) if prsn_present else None,
    )


def require_nlst_available(local_root_env: str = DEFAULT_LOCAL_ROOT_ENV) -> NLSTAvailability:
    """
    Same as check_nlst_availability, but raises NLSTAccessError with an
    actionable message (env var to set, DUA steps) when NLST is not
    actually available — for a caller that must fail loudly (e.g. a real
    NLST-outcome ingestion command) rather than silently produce an
    incomplete dataset.
    """
    status = check_nlst_availability(local_root_env)
    if not status.available:
        raise NLSTAccessError(
            f"NLST data is not available in this environment (checked env var "
            f"{local_root_env!r} = {status.local_root!r}). NLST is controlled-access: "
            "an approved NCI Data Use Agreement is required before any participant-level "
            "file may be used. See src/data/downloaders.py::print_nlst_instructions for "
            f"the authorized-access steps. If you have approved access, set {local_root_env} "
            "to the local directory containing screen.csv and prsn.csv with the required "
            f"columns ({REQUIRED_SCREEN_COLUMNS}, {REQUIRED_PRSN_COLUMNS})."
        )
    return status
