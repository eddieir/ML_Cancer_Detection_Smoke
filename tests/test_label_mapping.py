"""data/label_mapping.py — deterministic contiguous effective smoke-label space."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.label_mapping import (
    EffectiveLabelMapping, build_effective_label_mapping, identity_label_mapping,
)
from constants import SMOKE_TYPES


def _report(affected_classes: dict, policy: str = "merge_into_dual_use_or_other") -> dict:
    return {"policy": policy, "affected_classes": affected_classes}


# ─── No-merge six-class case ──────────────────────────────────────────────────

def test_identity_mapping_is_six_classes_contiguous():
    m = identity_label_mapping()
    assert m.k == 6
    assert m.class_names == list(SMOKE_TYPES.values())
    assert m.raw_to_effective == {i: i for i in range(6)}


def test_build_mapping_with_no_affected_classes_is_identity():
    m = build_effective_label_mapping(_report({}))
    assert m.k == 6
    assert m.excluded_raw_ids == []
    assert m.merged_away_raw_ids == []


def test_build_mapping_kept_with_warning_stays_six_classes():
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "kept_with_warning"}},
                      policy="keep_with_warning")
    m = build_effective_label_mapping(report)
    assert m.k == 6
    assert m.merged_away_raw_ids == []


# ─── Merged-class case ─────────────────────────────────────────────────────────

def test_merged_class_produces_contiguous_five_class_mapping():
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "merged_into_dual_use"}})
    m = build_effective_label_mapping(report)
    assert m.k == 5
    # ids are contiguous 0..4
    assert sorted(m.effective_id_to_name) == [0, 1, 2, 3, 4]
    assert m.merged_away_raw_ids == [2]  # cigar's raw id
    assert "cigar" not in m.class_names


def test_excluded_class_produces_contiguous_five_class_mapping():
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "excluded"}},
                      policy="exclude_from_training_and_evaluation")
    m = build_effective_label_mapping(report)
    assert m.k == 5
    assert m.excluded_raw_ids == [2]
    assert "cigar" not in m.class_names


def test_no_dead_merged_away_output_remains():
    """The merge target itself (dual_use) must still be reachable, and the
    merged-away raw id must not appear anywhere in the effective space."""
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "merged_into_dual_use"}})
    m = build_effective_label_mapping(report)
    assert "dual_use" in m.class_names
    assert 2 not in m.raw_to_effective  # cigar's raw id has no effective slot
    assert len(set(m.raw_to_effective.values())) == m.k  # no duplicate effective ids


def test_transform_maps_raw_to_contiguous_effective_ids():
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "merged_into_dual_use"}})
    m = build_effective_label_mapping(report)
    # raw ids 0,1,3,4,5 are active (2=cigar merged away)
    raw_ids = np.array([0, 1, 3, 4, 5])
    effective = m.transform(raw_ids)
    assert effective.max() == m.k - 1
    assert effective.min() == 0
    assert len(set(effective.tolist())) == 5


def test_transform_rejects_excluded_raw_id():
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "excluded"}},
                      policy="exclude_from_training_and_evaluation")
    m = build_effective_label_mapping(report)
    with pytest.raises(ValueError):
        m.transform(np.array([2]))  # cigar's raw id — must have been filtered out already


def test_inverse_transform_round_trips():
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "merged_into_dual_use"}})
    m = build_effective_label_mapping(report)
    raw_ids = np.array([0, 1, 3, 4, 5])
    effective = m.transform(raw_ids)
    back = m.inverse_transform(effective)
    assert sorted(back.tolist()) == sorted(raw_ids.tolist())


# ─── Save/load preserves class names and predictions ─────────────────────────

def test_to_dict_from_dict_round_trip_preserves_everything():
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "merged_into_dual_use"}})
    m = build_effective_label_mapping(report)
    loaded = EffectiveLabelMapping.from_dict(m.to_dict())
    assert loaded.k == m.k
    assert loaded.class_names == m.class_names
    assert loaded.raw_to_effective == m.raw_to_effective
    assert loaded.effective_id_to_name == m.effective_id_to_name
    assert loaded.policy == m.policy


def test_round_trip_preserves_predictions():
    """A prediction made with the ORIGINAL mapping's effective ids must be
    interpretable identically after saving/loading the mapping."""
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "merged_into_dual_use"}})
    m = build_effective_label_mapping(report)
    loaded = EffectiveLabelMapping.from_dict(m.to_dict())
    for eff_id in range(m.k):
        assert m.class_names[eff_id] == loaded.class_names[eff_id]


# ─── Incompatible metadata rejected ───────────────────────────────────────────

def test_validate_compatible_passes_for_identical_mappings():
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "merged_into_dual_use"}})
    m1 = build_effective_label_mapping(report)
    m2 = build_effective_label_mapping(report)
    m1.validate_compatible(m2)  # no raise


def test_validate_compatible_rejects_different_active_class_sets():
    """Two policy runs that excluded DIFFERENT raw classes end up with
    genuinely different effective label spaces, even though both have K=5 —
    validate_compatible must catch this, not just compare K."""
    cigar_excluded = _report(
        {"cigar": {"n_subjects": 1, "too_rare": True, "action": "excluded"}},
        policy="exclude_from_training_and_evaluation",
    )
    cannabis_excluded = _report(
        {"cannabis": {"n_subjects": 1, "too_rare": True, "action": "excluded"}},
        policy="exclude_from_training_and_evaluation",
    )
    m1 = build_effective_label_mapping(cigar_excluded)
    m2 = build_effective_label_mapping(cannabis_excluded)
    assert m1.k == m2.k == 5
    with pytest.raises(ValueError):
        m1.validate_compatible(m2)


def test_validate_compatible_rejects_identity_vs_merged():
    m1 = identity_label_mapping()
    report = _report({"cigar": {"n_subjects": 1, "too_rare": True, "action": "merged_into_dual_use"}})
    m2 = build_effective_label_mapping(report)
    with pytest.raises(ValueError):
        m1.validate_compatible(m2)
