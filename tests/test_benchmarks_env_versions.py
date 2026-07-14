"""benchmarks/env_versions.py — CI/reproducibility-artifact version lookups."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks import env_versions


def test_package_version_returns_none_for_an_uninstalled_distribution():
    assert env_versions.package_version("this-package-does-not-exist-xyz") is None


def test_package_version_returns_a_real_string_for_pytest():
    assert env_versions.package_version("pytest") == pytest.__version__


def test_scikit_learn_label_maps_to_the_correct_distribution_name():
    # The import module is `sklearn` but the installable distribution is
    # `scikit-learn` — CORE_PACKAGES must use the distribution name, not
    # the import name, or the lookup silently returns None for an installed
    # package (the exact bug this module fixes).
    assert env_versions.CORE_PACKAGES["scikit-learn"] == "scikit-learn"
    assert "sklearn" not in env_versions.CORE_PACKAGES.values()
    import sklearn

    assert env_versions.package_version("scikit-learn") == sklearn.__version__


def test_pyyaml_label_maps_to_the_correct_distribution_name():
    assert env_versions.CORE_PACKAGES["pyyaml"] == "PyYAML"
    import yaml

    assert env_versions.package_version("PyYAML") == yaml.__version__


def test_collect_core_package_versions_non_required_never_raises(monkeypatch):
    monkeypatch.setattr(env_versions, "package_version", lambda name: None)
    versions = env_versions.collect_core_package_versions(required=False)
    assert set(versions) == set(env_versions.CORE_PACKAGES)
    assert all(v is None for v in versions.values())


def test_collect_core_package_versions_required_raises_on_missing(monkeypatch):
    real = env_versions.package_version

    def fake(name):
        if name == "pytest":
            return None
        return real(name)

    monkeypatch.setattr(env_versions, "package_version", fake)
    with pytest.raises(RuntimeError, match="pytest"):
        env_versions.collect_core_package_versions(required=True)


def test_collect_core_package_versions_required_succeeds_when_all_installed():
    # Every package in CORE_PACKAGES is a real dependency of this project's
    # tested environment (see constraints-ci.txt) so this must not raise
    # when run against the actual test environment.
    versions = env_versions.collect_core_package_versions(required=True)
    assert all(v is not None for v in versions.values())
