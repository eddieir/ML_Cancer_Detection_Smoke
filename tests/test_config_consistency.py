"""
Phase 4 — configuration-consistency checks (Step 18).

Every safe default this hardening effort introduces or touches must agree
across configs/default.yaml, the Python code that actually applies the
default, and (where relevant) the CLI surface — a drift between "what the
YAML says" and "what the code actually does" is exactly the kind of gap
this file exists to catch (see the pre-existing missing_gene_policy bug
this effort fixed: declared in YAML, never read by the code, until now).
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─── Gene-contract policy (already covered directly in
#     test_preprocessing_artifact_contract.py; re-verified here as part of
#     the single canonical config-consistency suite) ───────────────────────

def test_gene_contract_defaults_match_code():
    from data.preprocessing import (
        DEFAULT_DUPLICATE_GENE_POLICY,
        DEFAULT_MINIMUM_GENE_COVERAGE,
        DEFAULT_MISSING_GENE_POLICY,
        DEFAULT_UNEXPECTED_GENE_POLICY,
    )

    pp = _load_config()["preprocessing"]
    assert pp["missing_gene_policy"] == DEFAULT_MISSING_GENE_POLICY
    assert pp["duplicate_gene_policy"] == DEFAULT_DUPLICATE_GENE_POLICY
    assert pp["unexpected_gene_policy"] == DEFAULT_UNEXPECTED_GENE_POLICY
    assert float(pp["minimum_gene_coverage"]) == DEFAULT_MINIMUM_GENE_COVERAGE


# ─── Batch-correction policy ────────────────────────────────────────────────

def test_batch_correction_defaults_match_code():
    from data.transforms import DEFAULT_ALLOW_TRANSDUCTIVE_HARMONY, DEFAULT_BATCH_CORRECTION_MODE

    bc = _load_config()["preprocessing"]["batch_correction"]
    assert bc["mode"] == DEFAULT_BATCH_CORRECTION_MODE
    assert bc["allow_transductive_harmony"] == DEFAULT_ALLOW_TRANSDUCTIVE_HARMONY


def test_batch_correction_default_artifact_status_is_disabled_matching_yaml_default():
    """The YAML default (mode='none') must produce an artifact whose
    batch_correction_status is 'disabled' — the two representations of
    "no batch correction happened" (config-level and artifact-level) must
    describe the same actual state."""
    from data.preprocessing import PreprocessingArtifact

    default_status = PreprocessingArtifact.__dataclass_fields__["batch_correction_status"].default
    assert default_status == "disabled"


# ─── Legacy-checkpoint / bundle-validation policy: CLI default vs. code
#     default (no YAML surface for these — see honest limitations) ─────────

def test_inference_cli_legacy_checkpoint_flag_defaults_to_false_matching_predictor_default():
    """inference.py's --allow-legacy-checkpoint flag and Predictor.
    from_config's unsafe_legacy_mode parameter must have the SAME default
    (False/reject) — a CLI default that silently disagreed with the
    function it calls would mean the documented "safe by default" claim in
    README doesn't actually hold for CLI users."""
    import inspect

    from inference import Predictor, _build_cli

    sig = inspect.signature(Predictor.from_config)
    assert sig.parameters["unsafe_legacy_mode"].default is False

    parser = _build_cli()
    default_map = {a.dest: a.default for a in parser._actions}
    assert default_map["allow_legacy_checkpoint"] is False


def test_bundle_load_defaults_to_rejecting_legacy_bundles():
    import inspect

    from benchmarks.bundle import load_and_validate_bundle

    sig = inspect.signature(load_and_validate_bundle)
    assert sig.parameters["allow_legacy"].default is False


def test_minimum_gene_coverage_conservative_and_explicit():
    """Safe-defaults requirement: minimum coverage must be conservative
    (full coverage required) and explicitly present in YAML, not implied."""
    pp = _load_config()["preprocessing"]
    assert "minimum_gene_coverage" in pp
    assert float(pp["minimum_gene_coverage"]) == 1.0


def test_weak_labels_disabled_by_default():
    cfg = _load_config()
    assert cfg["data"]["weak_labels"]["enabled"] is False


def test_tcga_bulk_disabled_by_default():
    cfg = _load_config()
    assert cfg["data"]["tcga"]["enabled"] is False


def test_experiment_mode_human_only_by_default():
    cfg = _load_config()
    assert cfg["data"]["experiment_mode"] == "human_only"


def test_assay_policy_default_matches_code():
    from constants import DEFAULT_ASSAY_POLICY

    cfg = _load_config()
    assert cfg["data"]["assay_policy"] == DEFAULT_ASSAY_POLICY


def test_bulk_sources_disabled_by_default_and_excludes_from_single_cell_lists():
    cfg = _load_config()
    bulk = cfg["data"]["bulk_sources"]
    assert bulk["enabled"] is False
    bulk_names = {row[2] for row in bulk["datasets"]}
    for path, *_ in cfg["data"]["microarray_sources"]:
        assert not any(name in path for name in bulk_names)


# ─── Phase 5 — pathway hierarchical MIL defaults ────────────────────────────

def test_pathway_hierarchical_mil_disabled_by_default():
    cfg = _load_config()
    phm = cfg["model"]["pathway_hierarchical_mil"]
    assert phm["enabled"] is False


def test_pathway_hierarchical_mil_yaml_defaults_match_python_dataclass():
    from pathway_hierarchical_mil import PathwayHierarchicalMILConfig

    cfg = _load_config()
    phm = dict(cfg["model"]["pathway_hierarchical_mil"])
    phm.pop("uncertainty", None)
    phm.pop("gene_modules", None)
    parsed = PathwayHierarchicalMILConfig.from_dict(phm)
    defaults = PathwayHierarchicalMILConfig()
    for key in phm:
        if hasattr(defaults, key):
            assert getattr(parsed, key) == getattr(defaults, key), key


def test_pathway_hierarchical_mil_gene_modules_default_requires_explicit_source():
    cfg = _load_config()
    gm = cfg["model"]["pathway_hierarchical_mil"]["gene_modules"]
    assert gm["path"] is None
    assert gm["allow_synthetic_modules"] is False
    assert gm["empty_module_policy"] == "error"
