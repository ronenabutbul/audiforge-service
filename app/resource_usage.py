"""What a conversion actually cost us in machine time.

Railway bills allocated vCPU-minutes and GB-minutes, not requests, so the
price of a conversion is the compute it occupies while it runs. Nothing
recorded that: the server logged the vision model's spend and called it the
cost, which on a run that never calls a model reads as free.

CPU comes from `getrusage`. Both engines run through `subprocess.run`, which
reaps the child, so the child's CPU lands in RUSAGE_CHILDREN - Audiveris and
homr are counted, not just the Python around them.

Two honest limits, worth knowing before anyone quotes a figure from here:

  * The counters are per-process, not per-request. Two conversions running at
    once each see the other's CPU. Audiveris is serialised by a lock, so the
    overlap is small, but a busy service reads high.
  * Memory is a high-water mark, not an average. Railway charges the average,
    so memory here is an upper bound and the derived cost is a ceiling.
"""

import resource
import sys
import time
from pathlib import Path

_CGROUP_PEAK = Path("/sys/fs/cgroup/memory.peak")
_CGROUP_CURRENT = Path("/sys/fs/cgroup/memory.current")


def _cpu_seconds() -> float:
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        usage = resource.getrusage(who)
        total += usage.ru_utime + usage.ru_stime
    return total


def _peak_memory_mb() -> float:
    """The container's high-water memory, in MB.

    cgroup v2 first: it sees the whole container, including a subprocess
    that has already exited. `ru_maxrss` is the fallback and it only knows
    about this process and its reaped children - on Linux it is in KB.
    """
    for path in (_CGROUP_PEAK, _CGROUP_CURRENT):
        try:
            return int(path.read_text().strip()) / (1024 * 1024)
        except (OSError, ValueError):
            continue
    peak = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    # ru_maxrss is KB on Linux and bytes on macOS. Production is Linux and
    # reaches this line only if the cgroup files are missing, but a figure
    # that is wrong by 1024x on a developer machine is worse than no figure.
    per_mb = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / per_mb


class Meter:
    """Measures one conversion. Use it as a context manager."""

    def __init__(self):
        self.cpu_seconds = 0.0
        self.wall_seconds = 0.0
        self.peak_memory_mb = 0.0
        self._cpu_start = 0.0
        self._wall_start = 0.0

    def __enter__(self) -> "Meter":
        self._cpu_start = _cpu_seconds()
        self._wall_start = time.monotonic()
        return self

    def __exit__(self, *exc) -> bool:
        self.cpu_seconds = round(_cpu_seconds() - self._cpu_start, 3)
        self.wall_seconds = round(time.monotonic() - self._wall_start, 3)
        self.peak_memory_mb = round(_peak_memory_mb(), 1)
        return False

    def as_dict(self) -> dict:
        return {
            "cpu_seconds": self.cpu_seconds,
            "wall_seconds": self.wall_seconds,
            "peak_memory_mb": self.peak_memory_mb,
        }
