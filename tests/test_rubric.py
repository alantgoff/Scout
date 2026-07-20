"""Registry invariants for scout.rubric — the scorecard definitions must stay
internally consistent (weights sum, unique keys, complete anchors) or the
scoring math and the classifier prompt silently drift."""

from __future__ import annotations

import pytest

from scout import rubric


@pytest.mark.parametrize("rb", rubric.RUBRICS.values(), ids=lambda r: r.key)
def test_section_weights_sum_to_100(rb: rubric.Rubric) -> None:
    assert sum(s.weight for s in rb.sections) == pytest.approx(100)


@pytest.mark.parametrize("rb", rubric.RUBRICS.values(), ids=lambda r: r.key)
def test_criterion_weights_sum_to_one_per_section(rb: rubric.Rubric) -> None:
    for section in rb.sections:
        total = sum(c.weight for c in section.criteria)
        assert total == pytest.approx(1.0), f"{rb.key}/{section.key} = {total}"


@pytest.mark.parametrize("rb", rubric.RUBRICS.values(), ids=lambda r: r.key)
def test_criterion_keys_unique_within_rubric(rb: rubric.Rubric) -> None:
    keys = [c.key for c in rb.criteria]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("rb", rubric.RUBRICS.values(), ids=lambda r: r.key)
def test_section_keys_unique_within_rubric(rb: rubric.Rubric) -> None:
    keys = [s.key for s in rb.sections]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("rb", rubric.RUBRICS.values(), ids=lambda r: r.key)
def test_every_criterion_has_purpose_and_three_anchors(rb: rubric.Rubric) -> None:
    for c in rb.criteria:
        assert c.purpose.strip(), c.key
        assert len(c.anchors) == 3, c.key
        assert all(a.strip() for a in c.anchors), c.key


def test_founder_keys_shared_across_rubrics() -> None:
    # A customer_type correction must keep founder scores meaningful, so the
    # founders & cap table criterion keys are intentionally identical across
    # both rubrics (only the DNA criterion differs).
    b2b = {c.key for c in rubric.B2B_RUBRIC.sections[0].criteria}
    b2c = {c.key for c in rubric.B2C_RUBRIC.sections[0].criteria}
    shared = b2b & b2c
    assert {"prev_founder_experience", "investor_quality", "cap_table_risk",
            "capital_and_stage", "funding_velocity"} <= shared


def test_rubric_for_routes_customer_types() -> None:
    assert rubric.rubric_for("b2c").key == "b2c"
    for ct in ("b2b", "b2b2c", "mixed", None):
        assert rubric.rubric_for(ct).key == "b2b"


def test_default_section_weights_matches_registry() -> None:
    weights = rubric.default_section_weights()
    assert set(weights) == {"b2b", "b2c"}
    for rb in rubric.RUBRICS.values():
        assert weights[rb.key] == {s.key: s.weight for s in rb.sections}
    # Fresh dict each call — pydantic default factories must not share state.
    assert rubric.default_section_weights() is not weights


def test_band_boundaries() -> None:
    assert rubric.band_for(80) == "strong"
    assert rubric.band_for(79.9) == "promising"
    assert rubric.band_for(60) == "promising"
    assert rubric.band_for(59.9) == "weak"
    assert rubric.band_for(0) == "weak"


def test_prompt_block_contains_every_criterion_key() -> None:
    block = rubric.prompt_block()
    for rb in rubric.RUBRICS.values():
        for c in rb.criteria:
            assert f"- {c.key}:" in block, c.key
    assert "ENTERPRISE READINESS SCORECARD" in block
    assert "CONSUMER READINESS SCORECARD" in block
