from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class AllCallsFailedError(Exception):
    """Raised when every call in fan_out fails."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors: list[BaseException] = errors
        super().__init__(f"All {len(errors)} calls failed")


async def fan_out(
    providers: list[Any],
    call: Callable[[Any], Awaitable[T]],
    *,
    calls_per_provider: int = 2,
    timeout_s: float = 240.0,
) -> list[T]:
    """Run *calls_per_provider* calls against each provider, collect successes.

    Parameters
    ----------
    providers:
        Iterable of provider objects (each passed to *call*).
    call:
        Async function ``call(provider) -> T`` representing one extraction attempt.
    calls_per_provider:
        How many times to call *call* for each provider.
    timeout_s:
        Per-call timeout in seconds.

    Returns
    -----
    list[T]
        All successful results (may contain duplicates from multiple calls).

    Raises
    ------
    AllCallsFailedError
        If zero calls succeeded. Timeouts count as failures and are included
        in the error list.
    """
    tasks_with_provider: list[tuple[Any, asyncio.Task[T]]] = []
    for p in providers:
        for _ in range(calls_per_provider):
            task = asyncio.wait_for(call(p), timeout_s)
            tasks_with_provider.append((p, task))

    raw_results = await asyncio.gather(
        *(t for _, t in tasks_with_provider), return_exceptions=True
    )

    successes: list[T] = []
    all_errors: list[BaseException] = []

    for (p, _), result in zip(tasks_with_provider, raw_results):
        if isinstance(result, asyncio.TimeoutError):
            all_errors.append(result)
            provider_name = getattr(p, "name", "unknown")
            logger.warning(
                "Call timed out after %.2fs for provider '%s'", timeout_s, provider_name
            )
        elif isinstance(result, BaseException):
            all_errors.append(result)
            provider_name = getattr(p, "name", "unknown")
            logger.warning(
                "Call failed for provider '%s': %s", provider_name, result
            )
        else:
            successes.append(result)

    if not successes:
        raise AllCallsFailedError(errors=all_errors)

    return successes
