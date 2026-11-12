"""Data collection with CaP-X: run spec -> preflight -> ``run.sh`` that calls CaP-X's own ``envs/launch.py``.

The baseline's server settings are the defaults (``settings``), each cited. The generated ``sources.yaml`` is the
input of ``geniac-baseline convert rows``.
"""

from geniac_baseline.collect.plan import RunSpec, TaskSpec, load_spec, preflight, write_plan
from geniac_baseline.collect.preflight import PreflightError

__all__ = ["PreflightError", "RunSpec", "TaskSpec", "load_spec", "preflight", "write_plan"]
