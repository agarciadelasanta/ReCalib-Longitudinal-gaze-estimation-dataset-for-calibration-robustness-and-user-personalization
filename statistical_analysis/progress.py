"""Terminal progress scoped to a CLI run; library calls remain quiet."""

from contextlib import contextmanager
from contextvars import ContextVar
import sys

from tqdm import tqdm


_enabled = ContextVar("analysis_progress_enabled", default=False)


@contextmanager
def progress_enabled(enabled=True):
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)


def progress(iterable, description, *, total=None, unit="step"):
    if not _enabled.get():
        yield from iterable
        return
    with tqdm(
        iterable, desc=description, total=total, unit=unit,
        dynamic_ncols=True, leave=False,
    ) as bar:
        yield from bar


def stage(message):
    if _enabled.get():
        tqdm.write(message, file=sys.stderr)
