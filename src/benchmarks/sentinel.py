"""
benchmarks/sentinel.py — frozen-test-data access sentinels (Step 11).

test_guard.py already provides a durable, atomically-acquired one-time
guard around WHEN the frozen test split may be evaluated. This module adds
a complementary, independently checkable property: an object standing in
for frozen test data (expression values, labels, gene names, batch
metadata, or an entire bag/dataset) that raises the instant anything
actually tries to READ it — attribute access, iteration, indexing, len(),
numpy/pandas conversion, even a plain repr() — rather than merely being
undocumented-but-technically-reachable. Wrapping real test data in a
FrozenAccessSentinel and running the full development pipeline against it
turns "the dev pipeline never touches test data before the guard" from a
code-review claim into something a test can assert directly: if ANY
sentinel access is triggered, the test fails immediately with a traceback
pointing at the exact call site that touched it.

This is NOT a replacement for FrozenTestGuard — the guard controls WHEN
evaluate_frozen_test() (the one sanctioned test-touching function) may run
at all, even given real data. Sentinels controls WHETHER any other code
path accidentally reads test data at all, even without evaluating it.
"""

from typing import Any


class FrozenDataAccessError(RuntimeError):
    """Raised the instant any code attempts to actually use a
    FrozenAccessSentinel's wrapped value — see the module docstring."""


class FrozenAccessSentinel:
    """Wraps `label` (only used in the error message — the real value, if
    any, is never stored or returned). Every dunder a real test-data object
    (an AnnData, a numpy array, a pandas DataFrame/Series, a plain dict/
    list of subject/label metadata, a CellLevelDataset/SubjectLevelDataset)
    could plausibly be read through raises FrozenDataAccessError instead of
    performing the operation."""

    def __init__(self, label: str = "frozen test data"):
        object.__setattr__(self, "_sentinel_label", label)

    def _raise(self, action: str):
        raise FrozenDataAccessError(
            f"Attempted to {action} on a FrozenAccessSentinel standing in for "
            f"{object.__getattribute__(self, '_sentinel_label')} — this data must not be read "
            "before the frozen-test guard (benchmarks/test_guard.py) has been acquired. If this "
            "sentinel appears in a real development-only code path, that path is reading test "
            "data too early."
        )

    # Attribute access covers every named field/method a real test-data
    # object could expose (gene names, batch metadata, labels, .values,
    # .to_pandas(), .X, .obs, ...) through one implementation.
    def __getattr__(self, name: str) -> Any:
        self._raise(f"access attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        self._raise(f"set attribute {name!r}")

    def __delattr__(self, name: str) -> None:
        self._raise(f"delete attribute {name!r}")

    def __iter__(self):
        self._raise("iterate over")

    def __next__(self):
        self._raise("advance an iterator over")

    def __getitem__(self, key):
        self._raise(f"index with [{key!r}]")

    def __setitem__(self, key, value):
        self._raise(f"assign to index [{key!r}] of")

    def __len__(self) -> int:
        self._raise("call len() on")

    def __contains__(self, item) -> bool:
        self._raise(f"check containment of {item!r} in")

    def __array__(self, dtype=None):
        self._raise("convert to a numpy array")

    def __bool__(self) -> bool:
        self._raise("convert to bool")

    def __int__(self) -> int:
        self._raise("convert to int")

    def __float__(self) -> float:
        self._raise("convert to float")

    def __repr__(self) -> str:
        self._raise("repr()")

    def __str__(self) -> str:
        self._raise("str()")

    def __eq__(self, other) -> bool:
        self._raise("compare (==)")

    def __hash__(self):
        self._raise("hash()")

    def __call__(self, *args, **kwargs):
        self._raise("call")


def wrap_frozen(value: Any, label: str = "frozen test data") -> FrozenAccessSentinel:
    """Convenience constructor — `value` is intentionally discarded (never
    stored), so there is no path by which the wrapped sentinel could ever
    leak the real value even via introspection (e.g. vars(), __dict__)."""
    del value
    return FrozenAccessSentinel(label=label)
