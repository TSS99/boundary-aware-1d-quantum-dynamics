"""Circuit constructions for boundary-aware split-operator propagation.

Two transform families, distinguished by the boundary condition they represent:

``qft``
    Periodic (ring) topology.  Acts on the ``n_q`` data qubits directly, no
    ancillas.

``qst``
    Dirichlet (hard-wall) topology.  Realises the exact orthonormal DST-II via
    an odd extension on a ``4N``-point register, costing two ancillas which are
    returned to ``|0>`` unitarily.
"""

from __future__ import annotations
