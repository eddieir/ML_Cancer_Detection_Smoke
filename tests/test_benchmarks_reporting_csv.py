"""
Regression test: write_csv_table must handle rows with heterogeneous keys
(e.g. neural adapter fold records carry cell-capping fields baseline fold
records don't) without crashing — found while running the full smoke-task
CLI verification with --leave-one-source-out and the neural model together.
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.reporting import write_csv_table


def test_write_csv_table_handles_heterogeneous_row_keys():
    rows = [
        {"model": "majority", "seed": 42, "fold": 0},
        {"model": "neural", "seed": 42, "fold": 0, "feature_mode": "cell_capped", "max_cells_per_subject": 5},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.csv"
        write_csv_table(path, rows)
        with open(path) as f:
            read_rows = list(csv.DictReader(f))
        assert read_rows[0]["feature_mode"] == ""  # restval for a row missing this key
        assert read_rows[1]["feature_mode"] == "cell_capped"
        assert read_rows[1]["max_cells_per_subject"] == "5"
