"""Boundary-aware spectral split-operator propagation for 1D quantum dynamics.

The organising idea of this package is that the spectral transform used inside a
split-operator step is part of the physical model, because it fixes the boundary
topology the propagator represents: a DFT/QFT represents a ring, a DST/QST
represents a box with hard walls.
"""

from __future__ import annotations

__version__ = "0.1.0"
