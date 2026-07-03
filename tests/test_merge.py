from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from doc_extractor.merge import merge_field, merge_extractions, normalize
from doc_extractor.models import (
    ExtractedField,
    FieldHit,
    FieldSpec,
    MergeGroup,
    MergeGroups,
    OutputSchema,
    SchemaExtraction,
    ValueGroup,
)
from doc_extractor.provider import OllamaProvider


class FakeProvider:
    """Mock merge provider that returns pre-programmed MergeGroups."""

    def __init__(self, return_value: MergeGroups | None = None):
        self.return_value = return_value
        self.structured_calls: list[tuple[str, type]] = []

    async def structured(self, prompt: str, response_model: type) -> MergeGroups:
        self.structured_calls.append((prompt, response_model))
        if self.return_value is None:
            raise RuntimeError("FakeProvider: structured() called with no return_value set")
        return self.return_value


def _make_hit(name: str, value: str | None, grounding: str | None = None) -> FieldHit:
    return FieldHit(name=name, value=value, grounding=grounding)


def _make_spec(name: str) -> FieldSpec:
    return FieldSpec(name=name, description=f"Field {name}")


# ---- (a) 4 identical values -> no LLM call, score 1.0 ----

@pytest.mark.asyncio
async def test_identical_values_skip_llm():
    provider = FakeProvider(return_value=None)  # should never be called
    spec = _make_spec("publisher")
    hits = [_make_hit("publisher", "McKinsey Global Institute") for _ in range(4)]
    total_calls = 4

    result = await merge_field(
        name="publisher",
        spec=spec,
        hits=hits,
        total_calls=total_calls,
        merge_provider=provider,
    )

    assert len(provider.structured_calls) == 0, "LLM should not be called for single pre-group"
    assert result.value == "McKinsey Global Institute"
    assert result.count == 4
    assert result.score == 1.0
    assert len(result.candidates) == 1
    assert result.candidates[0].canonical_value == "McKinsey Global Institute"


# ---- (b) variants grouped by LLM -> count 3, canonical 'McKinsey Global Institute' ----

@pytest.mark.asyncio
async def test_llm_groups_variants():
    # LLM groups all variants into one semantic group
    # hits: 'MGI' x1, 'McKinsey Global Institute' x2 -> 2 pre-groups, both merged by LLM
    provider = FakeProvider(
        return_value=MergeGroups(
            groups=[
                MergeGroup(
                    canonical_value="McKinsey Global Institute",
                    variants=["MGI", "McKinsey Global Institute"],
                )
            ]
        )
    )
    spec = _make_spec("publisher")
    hits = [
        _make_hit("publisher", "MGI"),
        _make_hit("publisher", "McKinsey Global Institute"),
        _make_hit("publisher", "McKinsey Global Institute"),
    ]
    total_calls = 3

    result = await merge_field(
        name="publisher",
        spec=spec,
        hits=hits,
        total_calls=total_calls,
        merge_provider=provider,
    )

    assert result.value == "McKinsey Global Institute"
    assert result.count == 3
    assert result.score == pytest.approx(3 / 3)
    assert len(result.candidates) == 1
    assert result.candidates[0].canonical_value == "McKinsey Global Institute"


# ---- (c) LLM returns groups missing a variant -> fallback, warning logged (caplog) ----

@pytest.mark.asyncio
async def test_llm_missing_variant_fallback(caplog):
    # Input has 4 distinct normalized variants, but LLM omits "Other Corp"
    provider = FakeProvider(
        return_value=MergeGroups(
            groups=[
                MergeGroup(canonical_value="MGI", variants=["MGI", "McKinsey Global Institute"]),
                MergeGroup(canonical_value="$50B", variants=["$50B", "Other Corp"]),  # invented "Other Corp" but misses some inputs
            ]
        )
    )
    spec = _make_spec("publisher")
    hits = [
        _make_hit("publisher", "MGI"),
        _make_hit("publisher", "McKinsey Global Institute"),
        _make_hit("publisher", "$50B"),
        _make_hit("publisher", "Walmart"),
    ]
    total_calls = 4

    result = await merge_field(
        name="publisher",
        spec=spec,
        hits=hits,
        total_calls=total_calls,
        merge_provider=provider,
    )

    assert "fallback to pre-groups" in caplog.text.lower() or "lost or invented" in caplog.text.lower()
    # Fallback: all 4 distinct pre-groups stay as-is
    assert len(result.candidates) == 4
    all_variants = set()
    for c in result.candidates:
        all_variants.update(c.variants)
    assert all_variants == {"MGI", "McKinsey Global Institute", "$50B", "Walmart"}


# ---- (d) 3 null + 1 value -> winner None with count 3 ----

@pytest.mark.asyncio
async def test_null_majority():
    provider = FakeProvider(return_value=None)  # not called: only 1 distinct non-null
    spec = _make_spec("publisher")
    hits = [
        _make_hit("publisher", None),
        _make_hit("publisher", None),
        _make_hit("publisher", None),
        _make_hit("publisher", "SomeValue"),
    ]
    total_calls = 4

    result = await merge_field(
        name="publisher",
        spec=spec,
        hits=hits,
        total_calls=total_calls,
        merge_provider=provider,
    )

    assert result.value is None
    assert result.count == 3
    assert result.score == pytest.approx(3 / 4)


# ---- (e) merge_extractions: absent field -> score 0.0, non-schema field ignored ----

@pytest.mark.asyncio
async def test_merge_extractions_absent_field_and_ignore_unknown():
    provider = FakeProvider(return_value=MergeGroups(groups=[MergeGroup(canonical_value="McKinsey", variants=["McKinsey"])])
    )
    schema = OutputSchema(
        schema_id="test",
        fields=[
            FieldSpec(name="publisher", description="Publisher org"),
            FieldSpec(name="author", description="Report author"),  # no hits for this
        ],
    )
    extractions = [
        SchemaExtraction(
            fields=[
                _make_hit("publisher", "McKinsey"),
                _make_hit("publisher", "mckinsey"),
                _make_hit("unknown_field", "garbage"),  # ignored
            ]
        ),
        SchemaExtraction(
            fields=[
                _make_hit("publisher", "McKinsey Global Institute"),
            ]
        ),
    ]
    total_calls = len(extractions)

    results = await merge_extractions(
        schema=schema,
        extractions=extractions,
        merge_provider=provider,
    )

    assert len(results) == 2
    # publisher should have hits merged
    pub = results[0]
    assert pub.name == "publisher"
    assert pub.count > 0
    assert pub.total_calls == total_calls

    # author should have zero hits -> value=None, count=0, score=0.0
    author = results[1]
    assert author.name == "author"
    assert author.value is None
    assert author.count == 0
    assert author.score == 0.0
    assert len(author.candidates) == 0
