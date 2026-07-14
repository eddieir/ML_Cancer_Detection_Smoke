"""benchmarks/env_versions.py — canonical distribution-name lookups for the
core scientific/runtime/test dependencies this project's behavior and test
suite depend on.

Shared by two callers that must never silently disagree or silently paper
over a missing package:
  - CI's "Record environment" workflow step (`python -m benchmarks.env_versions`,
    see .github/workflows/tests.yml), which must FAIL the job if a required
    package is not actually installed rather than printing a placeholder;
  - benchmarks/reporting.py's `_environment_snapshot`, which records a
    reproducibility artifact for a benchmark run and tolerates a missing
    optional package as `None` (a run can be inspected even if, say, torch
    was never installed for a classical-only benchmark).
"""
from importlib.metadata import PackageNotFoundError, version

# label -> installable distribution name. Not always the same as the
# import module name — e.g. `import sklearn` is distributed as
# `scikit-learn`, and `import yaml` is distributed as `PyYAML`.
CORE_PACKAGES = {
    "pytest": "pytest",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
    "torch": "torch",
    "scanpy": "scanpy",
    "anndata": "anndata",
    "celltypist": "celltypist",
    "harmonypy": "harmonypy",
    "pyyaml": "PyYAML",
}


def package_version(distribution_name: str):
    """Returns the installed version string for a distribution, or None if
    it is not installed. Never guesses or fabricates a version."""
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def collect_core_package_versions(required: bool = True) -> dict:
    """Returns {label: version_string} for every package in CORE_PACKAGES.

    If required is True (the CI use case), raises RuntimeError naming every
    package that is not actually installed instead of recording a
    placeholder — a broken environment must fail the job loudly, not print
    'UNAVAILABLE' and continue running tests it may not actually have.
    """
    versions = {label: package_version(dist) for label, dist in CORE_PACKAGES.items()}
    if required:
        missing = [label for label, v in versions.items() if v is None]
        if missing:
            raise RuntimeError(
                "Required package(s) not installed: " + ", ".join(sorted(missing))
            )
    return versions


def _main() -> None:
    import platform
    import sys

    print("python", sys.version.split()[0])
    print("platform", platform.platform())
    try:
        import pip

        print("pip", pip.__version__)
    except Exception:
        pass
    for label, ver in sorted(collect_core_package_versions(required=True).items()):
        print(label, ver)


if __name__ == "__main__":
    _main()
