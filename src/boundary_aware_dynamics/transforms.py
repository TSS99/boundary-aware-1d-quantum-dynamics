"""Spectral transforms and their exact conventions.

The transform is not an implementation detail: it fixes the boundary topology
the propagator represents.  A DFT/QFT represents a ring; a DST represents a box
with hard walls.  Both conventions are therefore pinned down explicitly here and
validated against closed-form expressions in ``tests/test_transforms.py``.

Discrete Fourier transform
    Forward (``numpy.fft.fft`` with ``norm="ortho"``)::

        F[k] = N^(-1/2) sum_j x[j] exp(-2 pi i j k / N)

    Frequencies are returned in ``numpy.fft.fftfreq`` bin order, i.e. bin ``k``
    carries signed index ``k_tilde = k`` for ``k < N/2`` and ``k - N`` otherwise.

    Qiskit's ``QFTGate`` implements the **opposite** exponential sign::

        QFT |j> = N^(-1/2) sum_k exp(+2 pi i j k / N) |k>

    so ``QFTGate`` corresponds to the *inverse* NumPy transform and the forward
    NumPy transform is ``QFTGate(...).inverse()``.  ``QFTGate`` includes the
    terminating swap network, so its output is in the same qubit ordering as its
    input and no manual bit reversal is required.  Basis state ``|j>`` is
    little-endian: qubit ``q`` carries bit ``2^q`` of ``j``.

Discrete sine transform, type II
    Forward (``scipy.fft.dst`` with ``type=2, norm="ortho"``) has the matrix

        S[nu - 1, j] = sqrt(2/N) sin(pi nu (j + 1/2) / N),    nu = 1 .. N-1
        S[N - 1,  j] = sqrt(1/N) sin(pi N (j + 1/2) / N)
                     = sqrt(1/N) (-1)^j

    Row ``nu - 1`` is exactly the continuum Dirichlet eigenfunction
    ``sqrt(2/L) sin(nu pi x / L)`` sampled on the midpoint grid and rescaled by
    ``sqrt(dx)``.  The final row carries the extra ``1/sqrt(2)`` because the
    ``nu = N`` mode is the Nyquist mode of the midpoint grid, where the sine
    samples to ``+-1`` and the usual orthogonality factor halves.

    ``S`` is real and orthogonal, so ``S^-1 = S^T`` and ``idst(..., type=2)``
    equals the DST-III of the same normalisation.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dst, idst

from .grids import validate_power_of_two


def fft_forward(psi: np.ndarray) -> np.ndarray:
    """Orthonormal forward DFT, output in ``fftfreq`` bin order."""
    return np.fft.fft(psi, norm="ortho")


def fft_inverse(psi_k: np.ndarray) -> np.ndarray:
    """Orthonormal inverse DFT."""
    return np.fft.ifft(psi_k, norm="ortho")


def dst2_forward(psi: np.ndarray) -> np.ndarray:
    """Orthonormal DST-II; output bin ``k`` carries sine mode ``nu = k + 1``."""
    return dst(psi, type=2, norm="ortho")


def dst2_inverse(psi_nu: np.ndarray) -> np.ndarray:
    """Orthonormal inverse DST-II (equivalently the orthonormal DST-III)."""
    return idst(psi_nu, type=2, norm="ortho")


def dft_matrix(n_grid: int) -> np.ndarray:
    """Return the explicit orthonormal forward DFT matrix."""
    validate_power_of_two(n_grid)
    j = np.arange(n_grid)
    return np.exp(-2j * np.pi * np.outer(j, j) / n_grid) / np.sqrt(n_grid)


def dst2_matrix(n_grid: int) -> np.ndarray:
    """Return SciPy's orthonormal DST-II matrix, obtained column by column."""
    validate_power_of_two(n_grid)
    return dst(np.eye(n_grid), type=2, norm="ortho", axis=0)


def analytical_dst2_matrix(n_grid: int) -> np.ndarray:
    """Return the DST-II matrix from its closed form, independently of SciPy.

    Used to validate the SciPy convention rather than merely reproducing it, and
    to give the circuit construction an analytical target.
    """
    validate_power_of_two(n_grid)
    nu = np.arange(1, n_grid + 1, dtype=float)[:, None]
    j = np.arange(n_grid, dtype=float)[None, :]
    matrix = np.sqrt(2.0 / n_grid) * np.sin(np.pi * nu * (j + 0.5) / n_grid)
    matrix[-1] /= np.sqrt(2.0)
    return matrix


def analytical_sine_mode(mode_index: int, n_grid: int, length: float) -> np.ndarray:
    """Return ``sqrt(2/L) sin(nu pi x / L)`` sampled on the Dirichlet midpoint grid.

    This is a *physical* sample array (quadrature-normalised for
    ``1 <= nu <= N-1``), not a row of the transform matrix; the two differ by
    ``sqrt(dx)``.
    """
    if not 1 <= mode_index <= n_grid:
        raise ValueError(f"mode_index must lie in 1..{n_grid}, received {mode_index}.")
    spacing = length / n_grid
    positions = (np.arange(n_grid, dtype=float) + 0.5) * spacing
    return np.sqrt(2.0 / length) * np.sin(mode_index * np.pi * positions / length)
