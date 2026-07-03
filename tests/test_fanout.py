from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest

from doc_extractor.fanout import AllCallsFailedError, fan_out


class FakeProvider:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def fake_provider_a() -> FakeProvider:
    return FakeProvider("provider_a")


@pytest.fixture
def fake_provider_b() -> FakeProvider:
    return FakeProvider("provider_b")


# ---- (a) 2 providers x 2 calls -> 4 results ----

@pytest.mark.asyncio
async def test_fan_out_two_providers_two_calls_returns_four(fake_provider_a, fake_provider_b):
    call_count = 0

    async def fake_call(p: FakeProvider) -> dict:
        nonlocal call_count
        call_count += 1
        return {"provider": p.name, "count": call_count}

    providers = [fake_provider_a, fake_provider_b]
    results = await fan_out(providers, fake_call, calls_per_provider=2)

    assert len(results) == 4
    assert call_count == 4


# ---- (b) one call raises -> 3 results returned, no exception ----

@pytest.mark.asyncio
async def test_fan_out_one_call_raises_returns_three(fake_provider_a, fake_provider_b):
    call_count = 0

    async def fake_call(p: FakeProvider) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ValueError("simulated failure")
        return {"provider": p.name, "count": call_count}

    results = await fan_out(
        [fake_provider_a, fake_provider_b], fake_call, calls_per_provider=2
    )

    assert len(results) == 3
    assert isinstance(results, list)


# ---- (c) all raise -> AllCallsFailedError with 4 errors ----

@pytest.mark.asyncio
async def test_fan_out_all_raise_raises_all_calls_failed(fake_provider_a, fake_provider_b):
    async def raise_call(p: FakeProvider) -> dict:
        raise ValueError(f"{p.name} failed")

    with pytest.raises(AllCallsFailedError) as exc_info:
        await fan_out(
            [fake_provider_a, fake_provider_b],
            raise_call,
            calls_per_provider=2,
        )

    assert len(exc_info.value.errors) == 4
    assert all(isinstance(e, ValueError) for e in exc_info.value.errors)


# ---- (d) slow call exceeding tiny timeout_s=0.01 -> AllCallsFailedError ----

@pytest.mark.asyncio
async def test_fan_out_slow_call_dropped_on_timeout(fake_provider_a: FakeProvider):
    async def slow_call(p: FakeProvider) -> dict:
        await asyncio.sleep(1)
        return {"ok": True}

    with pytest.raises(AllCallsFailedError) as exc_info:
        await fan_out(
            [fake_provider_a],
            slow_call,
            calls_per_provider=2,
            timeout_s=0.01,
        )

    assert len(exc_info.value.errors) == 2
    assert all(isinstance(e, asyncio.TimeoutError) for e in exc_info.value.errors)


# ---- (e) mix of success and timeout ----

@pytest.mark.asyncio
async def test_fan_out_mix_success_and_timeout(fake_provider_a, fake_provider_b):
    call_count = 0

    async def mixed_call(p: FakeProvider) -> dict:
        nonlocal call_count
        call_count += 1
        if p.name == "provider_b":
            await asyncio.sleep(1)
            return {"ok": True}
        return {"provider": p.name, "count": call_count}

    results = await fan_out(
        [fake_provider_a, fake_provider_b],
        mixed_call,
        calls_per_provider=2,
        timeout_s=0.05,
    )

    # provider_b 2 slow calls => 2 timeouts
    # provider_a 2 fast calls => 2 successes
    assert len(results) == 2


# ---- (f) all timeouts -> AllCallsFailedError ----

@pytest.mark.asyncio
async def test_all_timeouts_raise():
    async def slow_call(provider):
        await asyncio.sleep(10)

    with pytest.raises(AllCallsFailedError):
        await fan_out(["p1"], slow_call, calls_per_provider=2, timeout_s=0.01)
