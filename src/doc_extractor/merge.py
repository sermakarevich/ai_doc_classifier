from __future__ import annotations

import logging
from collections import defaultdict
from typing import NamedTuple

from .models import (
    ExtractedField,
    FieldHit,
    FieldSpec,
    Forecast,
    ForecastExtraction,
    MergedForecast,
    MergeGroup,
    MergeGroups,
    OutputSchema,
    SchemaExtraction,
    ValueGroup,
)
from .constants import FORECAST_VALUE_FIELDS
from .provider import OllamaProvider
from .prompts import merge_prompt

logger = logging.getLogger(__name__)


class MergedGroup(NamedTuple):
    canonical: str | None
    count: int
    variants: list[str]
    groundings: list[str]


def normalize(v: str) -> str:
    """Strip, casefold, collapse internal whitespace."""
    return " ".join(v.strip().casefold().split())


# ---- helpers ----

def _pick_winner(candidates: list[ValueGroup]) -> ValueGroup:
    """Ties: prefer non-null."""
    return max(candidates, key=lambda g: (g.count, int(g.canonical_value is not None)))


def _make_result(
    name: str, value: str | None, count: int, total_calls: int, candidates: list[ValueGroup]
) -> ExtractedField:
    return ExtractedField(
        name=name, value=value, count=count,
        total_calls=total_calls,
        score=count / total_calls if total_calls else 0.0,
        candidates=candidates,
    )


# ---- core algorithm ----

async def merge_field(
    name: str,
    spec: FieldSpec,
    hits: list[FieldHit],
    total_calls: int,
    merge_provider: OllamaProvider,
) -> ExtractedField:
    # Step 1: split into null and value hits
    value_hits = [h for h in hits if h.value is not None]
    null_hits = [h for h in hits if h.value is None]

    # Step 2: pre-group by normalized value
    # norm -> list of hits sharing that normalized form
    norm_hits: dict[str, list[FieldHit]] = defaultdict(list)
    norm_order: list[str] = []      # insertion order of distinct normalizations
    for h in value_hits:
        n = normalize(h.value)
        if n not in norm_hits:
            norm_hits[n] = []
            norm_order.append(n)
        norm_hits[n].append(h)

    # Step 3a: 0 value hits -> null group wins
    if not norm_order:
        groundings = [h.grounding for h in null_hits if h.grounding]
        candidates = [ValueGroup(canonical_value=None, count=len(null_hits), variants=[], groundings=groundings)]
        return _make_result(name, None, len(null_hits), total_calls, candidates)

    # Step 3b: exactly 1 distinct pre-group -> no LLM
    if len(norm_order) == 1:
        n = norm_order[0]
        hlist = norm_hits[n]
        rep = hlist[0].value
        candidates = [ValueGroup(canonical_value=rep, count=len(hlist), variants=[rep], groundings=[h.grounding for h in hlist if h.grounding])]
        if null_hits:
            candidates.append(ValueGroup(canonical_value=None, count=len(null_hits), variants=[], groundings=[h.grounding for h in null_hits if h.grounding]))
        winner = _pick_winner(candidates)
        return _make_result(name, winner.canonical_value, winner.count, total_calls, candidates)

    # Step 4: >1 pre-groups -> semantic merge via LLM
    raw_variants: list[str] = [norm_hits[n][0].value for n in norm_order]  # first-seen raw per pre-group
    merge_result: MergeGroups = await merge_provider.structured(
        merge_prompt(name, spec.description, raw_variants), MergeGroups
    )

    # Validate: every input variant must appear exactly once in LLM output
    output_variants: set[str] = set()
    for mg in merge_result.groups:
        output_variants.update(mg.variants)
    if output_variants != set(raw_variants):
        logger.warning(
            "merge_field '%s': LLM output lost or invented variants; falling back to pre-groups.", name
        )
        merged_items = _fallback_to_pre_groups(norm_order, norm_hits, null_hits)
    else:
        merged_items = _build_semantic_groups(
            raw_variants, norm_order, norm_hits, merge_result, null_hits
        )

    candidates = _merged_to_value_groups(merged_items)
    winner = _pick_winner(candidates)
    return _make_result(name, winner.canonical_value, winner.count, total_calls, candidates)


def _fallback_to_pre_groups(
    norm_order: list[str], norm_hits: dict[str, list[FieldHit]], null_hits: list[FieldHit]
) -> list[MergedGroup]:
    items: list[MergedGroup] = []
    for n in norm_order:
        hlist = norm_hits[n]
        items.append(MergedGroup(hlist[0].value, len(hlist), [hlist[0].value], [h.grounding for h in hlist if h.grounding]))
    if null_hits:
        items.append(MergedGroup(None, len(null_hits), [], [h.grounding for h in null_hits if h.grounding]))
    return items


def _build_semantic_groups(
    raw_variants: list[str],
    norm_order: list[str],
    norm_hits: dict[str, list[FieldHit]],
    merge_result: MergeGroups,
    null_hits: list[FieldHit],
) -> list[MergedGroup]:
    items: list[MergedGroup] = []
    for mg in merge_result.groups:
        member_raw_indices: list[int] = []
        for v in mg.variants:
            try:
                member_raw_indices.append(raw_variants.index(v))
            except ValueError:
                pass
        if not member_raw_indices:
            continue

        canonical = mg.canonical_value if mg.canonical_value in raw_variants else raw_variants[member_raw_indices[0]]
        count = 0
        grounds: list[str] = []
        variants: list[str] = []
        for idx in member_raw_indices:
            norm = norm_order[idx]
            hlist = norm_hits[norm]
            count += len(hlist)
            grounds.extend(h.grounding for h in hlist if h.grounding)
            variants.append(raw_variants[idx])
        items.append(MergedGroup(canonical, count, variants, grounds))

    if null_hits:
        items.append(MergedGroup(None, len(null_hits), [], [h.grounding for h in null_hits if h.grounding]))
    return items


def _merged_to_value_groups(
    items: list[MergedGroup]
) -> list[ValueGroup]:
    return [
        ValueGroup(canonical_value=g.canonical, count=g.count, variants=list(dict.fromkeys(g.variants)), groundings=list(dict.fromkeys(g.groundings)))
        for g in items
    ]


# ---- merge_extractions ----

async def merge_extractions(
    schema: OutputSchema,
    extractions: list[SchemaExtraction],
    merge_provider: OllamaProvider,
) -> list[ExtractedField]:
    # Case-insensitive hit collection
    hit_map: dict[str, list[FieldHit]] = {}
    for ext in extractions:
        for hit in ext.fields:
            key = hit.name.casefold()
            hit_map.setdefault(key, []).append(hit)

    total_calls = len(extractions)
    results: list[ExtractedField] = []

    for spec in schema.fields:
        key = spec.name.casefold()
        hits = hit_map.get(key, [])

        if not hits:
            results.append(
                ExtractedField(name=spec.name, value=None, count=0, total_calls=total_calls, score=0.0, candidates=[])
            )
        else:
            results.append(
                await merge_field(name=spec.name, spec=spec, hits=hits, total_calls=total_calls, merge_provider=merge_provider)
            )

    return results


# ---- forecast merging ----

def _majority_value(values: list[str]) -> str | None:
    """Most common value by normalized form; ties broken by first appearance."""
    if not values:
        return None
    counts: dict[str, int] = {}
    reps: dict[str, str] = {}
    for v in values:
        n = normalize(v)
        counts[n] = counts.get(n, 0) + 1
        reps.setdefault(n, v)
    best = max(counts, key=lambda n: counts[n])
    return reps[best]


async def merge_forecasts(
    extractions: list[ForecastExtraction],
    merge_provider: OllamaProvider,
) -> list[MergedForecast]:
    """Group forecasts from all calls by sector, majority-vote each field."""
    total_calls = len(extractions)

    # Pre-group by normalized sector name; remember first raw name and call index
    norm_members: dict[str, list[tuple[int, Forecast]]] = {}
    norm_order: list[str] = []
    reps: dict[str, str] = {}
    for idx, ext in enumerate(extractions):
        for f in ext.forecasts:
            if not f.sector_name or not f.sector_name.strip():
                continue
            n = normalize(f.sector_name)
            if n not in norm_members:
                norm_members[n] = []
                norm_order.append(n)
                reps[n] = f.sector_name
            norm_members[n].append((idx, f))

    if not norm_order:
        return []

    # Semantic merge of sector-name variants via LLM (reuses merge_prompt/MergeGroups)
    groups: list[list[str]]  # lists of norms
    canonicals: list[str]
    if len(norm_order) == 1:
        groups = [[norm_order[0]]]
        canonicals = [reps[norm_order[0]]]
    else:
        raw_variants = [reps[n] for n in norm_order]
        try:
            merge_result: MergeGroups = await merge_provider.structured(
                merge_prompt("sector_name", "name of the sector/market/arena being forecast", raw_variants),
                MergeGroups,
            )
            output_variants: set[str] = set()
            for mg in merge_result.groups:
                output_variants.update(mg.variants)
            if output_variants != set(raw_variants):
                raise ValueError("LLM lost or invented sector names")
            raw_to_norm = {reps[n]: n for n in norm_order}
            groups = []
            canonicals = []
            for mg in merge_result.groups:
                member_norms = [raw_to_norm[v] for v in mg.variants if v in raw_to_norm]
                if not member_norms:
                    continue
                groups.append(member_norms)
                canonicals.append(
                    mg.canonical_value if mg.canonical_value in raw_to_norm else reps[member_norms[0]]
                )
        except Exception as err:
            logger.warning("merge_forecasts: sector merge failed (%s); using pre-groups.", err)
            groups = [[n] for n in norm_order]
            canonicals = [reps[n] for n in norm_order]

    results: list[MergedForecast] = []
    for member_norms, canonical in zip(groups, canonicals):
        members = [m for n in member_norms for m in norm_members[n]]
        call_indices = {idx for idx, _ in members}
        count = len(call_indices)
        values = {
            field: _majority_value(
                [getattr(f, field) for _, f in members if getattr(f, field) is not None]
            )
            for field in FORECAST_VALUE_FIELDS
        }
        results.append(
            MergedForecast(
                sector_name=canonical,
                count=count,
                total_calls=total_calls,
                score=count / total_calls if total_calls else 0.0,
                **values,
            )
        )

    results.sort(key=lambda r: (-r.score, r.sector_name.casefold()))
    return results
