"""DSO Service 7 — AI-based Grid Topology Identification.

Two operational modes, per D2.2 §2.6.7:

  validate_topology(...)  — the main use case. Takes the DSO's documented
      topology plus measurements and either corrects it (when the set of
      physically possible connections is supplied) or ranks the documented
      edges by how suspicious they look (when it is not).

  reconstruct(...)        — blind reconstruction from measurements alone.
      Evaluated and found measurement-limited under realistic noise; kept
      as a capability and as a benchmark, not as the operational mode.
"""

from .blind import reconstruct
from .validation import validate_topology, flag_suspicious_edges
from .metrics import evaluate
from .distance import electrical_distance

__all__ = [
    "reconstruct",
    "validate_topology",
    "flag_suspicious_edges",
    "evaluate",
    "electrical_distance",
]
