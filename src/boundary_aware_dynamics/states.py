"""Initial states and the physical-sample / quantum-amplitude distinction.

Two different normalisations appear throughout this project and conflating them
is a common source of error, so they are kept explicit:

Physical samples
    ``psi(x_j)``, normalised in the quadrature norm ``dx * sum_j |psi(x_j)|^2 = 1``.
    This is the discretisation of a continuum wavefunction and carries units of
    ``length^(-1/2)``.

Quantum amplitudes
    ``|Psi> = sqrt(dx) * sum_j psi(x_j) |j>``, normalised in the Euclidean norm
    ``sum_j |a_j|^2 = 1``.  This is what a statevector simulator or a real
    register holds, and it is dimensionless.

Never hand quadrature-normalised physical samples to Qiskit directly; convert
with :func:`physical_to_amplitudes` first.

Gaussian convention
    ``psi(x, 0) ~ exp[-(x - x0)^2 / (4 sigma^2)] * exp[i k0 (x - x0)]``

so that ``|psi|^2 ~ exp[-(x - x0)^2 / (2 sigma^2)]`` and **sigma is the standard
deviation of the probability density**.  The same convention is used in the
README, the notebooks, the figure captions and the manuscript-alignment notes.
"""

from __future__ import annotations

import numpy as np


def quadrature_norm(psi: np.ndarray, spacing: float) -> float:
    """Return ``sqrt(dx * sum |psi|^2)``, the norm of physical samples."""
    return float(np.sqrt(np.real(spacing * np.vdot(psi, psi))))


def euclidean_norm(amplitudes: np.ndarray) -> float:
    """Return ``sqrt(sum |a|^2)``, the norm of quantum amplitudes."""
    return float(np.linalg.norm(amplitudes))


def normalise_physical(psi: np.ndarray, spacing: float) -> np.ndarray:
    """Rescale physical samples to unit quadrature norm."""
    norm = quadrature_norm(psi, spacing)
    if norm <= 0.0:
        raise ValueError("Cannot normalise a state with non-positive norm.")
    return psi.astype(np.complex128) / norm


def physical_to_amplitudes(psi: np.ndarray, spacing: float) -> np.ndarray:
    """Map physical samples to register amplitudes: ``a_j = sqrt(dx) psi(x_j)``."""
    if spacing <= 0:
        raise ValueError("spacing must be positive.")
    return np.sqrt(spacing) * psi.astype(np.complex128)


def amplitudes_to_physical(amplitudes: np.ndarray, spacing: float) -> np.ndarray:
    """Map register amplitudes back to physical samples: ``psi = a / sqrt(dx)``."""
    if spacing <= 0:
        raise ValueError("spacing must be positive.")
    return amplitudes.astype(np.complex128) / np.sqrt(spacing)


def probability_density(psi: np.ndarray) -> np.ndarray:
    """Return ``|psi|^2`` for physical samples (a density, integrates with ``dx``)."""
    return np.abs(psi) ** 2


def gaussian_wavepacket(
    positions: np.ndarray,
    spacing: float,
    centre: float,
    momentum: float,
    sigma: float,
) -> np.ndarray:
    """Return a normalised Gaussian wavepacket in the ``4 sigma^2`` convention.

    ``sigma`` is the standard deviation of ``|psi|^2``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    envelope = np.exp(-((positions - centre) ** 2) / (4.0 * sigma**2))
    carrier = np.exp(1j * momentum * (positions - centre))
    return normalise_physical(envelope * carrier, spacing)


def sine_windowed_gaussian(
    positions: np.ndarray,
    spacing: float,
    length: float,
    centre: float,
    momentum: float,
    sigma: float,
) -> np.ndarray:
    """Return a Gaussian multiplied by ``sin(pi x / L)`` so it vanishes at both walls.

    The window makes the state exactly representable in the Dirichlet sine basis,
    which is required for the hard-wall reference to be meaningful.
    """
    if not 0.0 <= centre <= length:
        raise ValueError("centre must lie inside the well interval [0, L].")
    packet = gaussian_wavepacket(positions, spacing, centre, momentum, sigma)
    return normalise_physical(packet * np.sin(np.pi * positions / length), spacing)


def align_global_phase(state: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate ``state`` by the global phase that maximises overlap with ``reference``.

    Statevector comparisons are only meaningful up to a global phase, so this
    must be applied before computing an L2 state error.  Fidelity is already
    phase-invariant and does not need it.
    """
    overlap = np.vdot(reference, state)
    if abs(overlap) < 1e-14:
        return state.astype(np.complex128)
    return state.astype(np.complex128) * np.conjugate(overlap) / abs(overlap)
