"""Opt-in fixed-cell atomic quench, separate from bulk relaxation."""

from numbers import Integral

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS

from .slab import SlabConfig, validate_slab


class SlabQuenchBoundsError(ValueError):
    """An optimizer move left the allowed slab geometry."""


class SlabQuenchNotConverged(RuntimeError):
    """The bounded atomic relaxation did not reach its force threshold."""


def quench_slab(atoms: Atoms, config: SlabConfig, calc_factory, *,
                fmax: float = 0.05, max_steps: int = 200) -> Atoms:
    """Return a valid, converged copy with a fresh ASE calculator attached.

    All xyz coordinates are free; the complete cell and physical PBC stay
    fixed. Input constraints are rejected. No wrapping or centering is done,
    so energies and forces refer to the returned coordinate frame. The factory
    must supply a fresh calculator supporting energy, forces and mixed PBC;
    stress is not requested. The zero-pressure SlabConfig contract applies.

    fmax is the maximum per-atom force norm in eV/Angstrom. max_steps bounds
    optimizer moves; zero only evaluates the input. Invalid geometry raises
    ValueError before evaluation, and nonfinite results or nonconvergence raise
    RuntimeError. Calculator errors propagate. The input is never relaxed in
    place. No connectivity or material-specific energy checks are performed.
    """
    if not np.isfinite(fmax) or fmax <= 0:
        raise ValueError("fmax must be finite and positive")
    if (isinstance(max_steps, bool) or not isinstance(max_steps, Integral)
            or max_steps < 0):
        raise ValueError("max_steps must be a nonnegative integer")
    validate_slab(atoms, config)
    if atoms.constraints:
        raise ValueError("Slab quench requires unconstrained atoms")

    relaxed = atoms.copy()
    cell = relaxed.cell.array.copy()
    calculator = calc_factory()
    if calculator is None or calculator is atoms.calc:
        raise ValueError("calc_factory must return a fresh calculator")
    relaxed.calc = calculator

    # A bounded step loop lets us validate geometry before the next calculation.
    # ASE versions differ in irun's evaluation order and handling of steps=0.
    with LBFGS(relaxed, logfile=None) as optimizer:
        for steps in range(max_steps + 1):
            try:
                validate_slab(relaxed, config)
            except ValueError as exc:
                raise SlabQuenchBoundsError(str(exc)) from exc
            if not np.array_equal(relaxed.cell.array, cell):
                raise ValueError("Slab cell changed during fixed-cell quench")
            energy = relaxed.get_potential_energy()
            forces = relaxed.get_forces()
            if (np.ndim(energy) != 0 or np.iscomplexobj(energy) or not np.isfinite(energy)
                    or forces.shape != (len(relaxed), 3)
                    or np.iscomplexobj(forces) or not np.isfinite(forces).all()):
                raise RuntimeError("Slab calculator must return finite real energy and (N, 3) forces")
            max_force = float(np.linalg.norm(forces, axis=1).max())
            if not np.isfinite(max_force):
                raise RuntimeError("Slab calculator returned a nonfinite force norm")
            if max_force < fmax:
                relaxed.info.update(slab_quench_steps=steps, slab_quench_fmax=max_force)
                return relaxed
            if steps < max_steps:
                optimizer.step()

    raise SlabQuenchNotConverged(
        f"Slab quench did not converge after {max_steps} steps: "
        f"max force {max_force:.6g} eV/Angstrom >= fmax {fmax:.6g}")
