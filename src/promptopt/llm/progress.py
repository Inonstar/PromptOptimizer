"""tqdm progress bars driven by client-level request events.

Every request an :class:`~promptopt.llm.base.LLMClient` issues can emit a
start/finish event; :class:`ProgressReporter` turns those into live tqdm
bars so concurrent evaluation shows progress (``judging 47/200``) instead of
going silent for minutes. Auto-disabled when stderr is not a TTY, so CI logs
stay clean.

:seealso: ``lib_structure.md`` -> "What backs each part" (tqdm)
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar, cast

from tqdm import tqdm

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


class ProgressReporter:
    """Creates tqdm bars, auto-disabled when output is not a terminal.

    :param enabled: ``None`` auto-detects via ``sys.stderr.isatty()``;
        ``True``/``False`` force the bar on/off.
    :type enabled: bool | None
    """

    def __init__(self, enabled: bool | None = None) -> None:
        """Initialize the reporter.

        :param enabled: ``None`` auto-detects via ``sys.stderr.isatty()``;
            ``True``/``False`` force the bar on/off.
        :type enabled: bool | None
        """
        if enabled is None:
            enabled = sys.stderr.isatty()
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Whether this reporter will actually render bars.

        :returns: ``True`` if bars render, ``False`` if every call is a no-op.
        :rtype: bool
        """
        return self._enabled

    def wrap(self, iterable: Iterable[ItemT], *, total: int | None = None, desc: str = "") -> Iterator[ItemT]:
        """Wrap an iterable with a progress bar as it is consumed.

        :param iterable: The iterable to wrap.
        :type iterable: Iterable[ItemT]
        :param total: Known length, if ``iterable`` doesn't support ``len()``.
        :type total: int | None
        :param desc: Label shown next to the bar.
        :type desc: str
        :returns: An iterator yielding the same items as ``iterable``.
        :rtype: Iterator[ItemT]
        """
        return iter(tqdm(iterable, total=total, desc=desc, disable=not self._enabled))

    def run_concurrent(
        self,
        fn: Callable[[ItemT], ResultT],
        items: list[ItemT],
        *,
        max_workers: int = 8,
        desc: str = "",
    ) -> list[ResultT]:
        """Run ``fn`` over ``items`` concurrently, reporting progress as each finishes.

        Results are returned in the same order as ``items``, regardless of
        completion order -- callers can zip the result back against its input.

        :param fn: The function to apply to each item. Must be thread-safe.
        :type fn: Callable[[ItemT], ResultT]
        :param items: The items to process.
        :type items: list[ItemT]
        :param max_workers: Maximum number of concurrent worker threads.
        :type max_workers: int
        :param desc: Label shown next to the bar.
        :type desc: str
        :returns: One result per item, in input order.
        :rtype: list[ResultT]
        :raises Exception: Whatever the first failing call to ``fn`` raised, once
            every in-flight call has finished.
        """
        if not items:
            return []
        # Slots are filled by index as each future completes; if any future raises,
        # `future.result()` re-raises here and the function exits before returning,
        # so a fully-`None` slot is never observable by a caller.
        results: list[ResultT | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(fn, item): index for index, item in enumerate(items)}
            bar = tqdm(total=len(items), desc=desc, disable=not self._enabled)
            try:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    bar.update(1)
            finally:
                bar.close()
        return cast(list[ResultT], results)
