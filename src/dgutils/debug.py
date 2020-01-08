import logging

from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DebugCounter:
    """Keep track of the number of images saved for debugging"""

    COUNTER: int = 0

    @property
    def count(self):
        _counter = self.COUNTER     # copy
        self.COUNTER += 1
        return _counter


DEBUGCOUNTER = DebugCounter()


def get_debug_path(identifier: str, *, mkdir: bool = True) -> Path:
    """Return a 'unique' path from  identifier based on epoch."""
    debug_path = Path("debug") / f"{identifier}{DEBUGCOUNTER.count}"
    logger.debug(debug_path)
    if mkdir:
        debug_path.mkdir(exist_ok=True, parents=True)
    return debug_path
