"""Opt-in physical slab quench, separate from bulk relaxation."""

from numbers import Integral

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS

from .slab import SlabConfig, validate_slab
from .slab_cell_relaxation import SlabCellRelaxation, _SlabCellFilter


class SlabQuenchBoundsError(ValueError):
    """An optimizer move left the allowed slab geometry."""


class SlabQuenchNotConverged(RuntimeError):
    """The bounded relaxation did not reach its physical thresholds."""


def quench_slab(atoms: Atoms, config: SlabConfig, calc_factory, *,
                fmax: float = 0.05, max_steps: int = 200,
                cell_relaxation: SlabCellRelaxation | None = None) -> Atoms:
    """Return a valid, converged copy with a fresh ASE calculator attached.

    All xyz coordinates are free. By default the complete cell stays fixed
    and stress is never requested. Input constraints are rejected. No wrapping
    or centering is done, so energies and forces refer to the returned frame.
    The factory must supply a fresh calculator supporting energy, forces and
    mixed PBC. The zero-pressure SlabConfig contract applies.

    Optional cell_relaxation also frees the xy cell lengths and shear, and
    requires ASE stress. Only the owned copy's cell is first canonicalized:
    tiny off-plane components become zero and c becomes (0, 0, cell_height),
    without scaling positions. The entire perpendicular c vector and physical
    PBC then stay fixed. Convergence additionally requires the maximum absolute
    in-plane stress times c to be below stress_max, in eV/Angstrom². No pressure
    term is added. LBFGS bounds steps in its combined atomic/deformation
    coordinates; an out-of-bounds slab proposal is rejected, not clipped.

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
    if cell_relaxation is not None and not isinstance(cell_relaxation, SlabCellRelaxation):
        raise TypeError("cell_relaxation must be SlabCellRelaxation or None")
    validate_slab(atoms, config)
    if atoms.constraints:
        raise ValueError("Slab quench requires unconstrained atoms")

    relaxed = atoms.copy()
    cell = relaxed.cell.array.copy()
    if cell_relaxation is not None:
        cell[:2, 2] = 0.0
        cell[2] = (0.0, 0.0, config.cell_height)
        relaxed.set_cell(cell, scale_atoms=False)
    calculator = calc_factory()
    if calculator is None or calculator is atoms.calc:
        raise ValueError("calc_factory must return a fresh calculator")
    relaxed.calc = calculator
    target = relaxed if cell_relaxation is None else _SlabCellFilter(relaxed)

    # A bounded step loop lets us validate geometry before the next calculation.
    # ASE versions differ in irun's evaluation order and handling of steps=0.
    with LBFGS(target, logfile=None) as optimizer:
        for steps in range(max_steps + 1):
            try:
                validate_slab(relaxed, config)
            except ValueError as exc:
                raise SlabQuenchBoundsError(str(exc)) from exc
            if cell_relaxation is None and not np.array_equal(relaxed.cell.array, cell):
                raise ValueError("Slab cell changed during fixed-cell quench")
            if cell_relaxation is not None and (
                    not np.array_equal(relaxed.cell[2], cell[2])
                    or np.any(relaxed.cell[:2, 2] != 0.0)):
                raise ValueError("Slab perpendicular cell changed during in-plane quench")
            energy = relaxed.get_potential_energy()
            forces = relaxed.get_forces()
            if (np.ndim(energy) != 0 or np.iscomplexobj(energy) or not np.isfinite(energy)
                    or forces.shape != (len(relaxed), 3)
                    or np.iscomplexobj(forces) or not np.isfinite(forces).all()):
                raise RuntimeError("Slab calculator must return finite real energy and (N, 3) forces")
            max_force = float(np.linalg.norm(forces, axis=1).max())
            if not np.isfinite(max_force):
                raise RuntimeError("Slab calculator returned a nonfinite force norm")
            stress_converged = True
            if cell_relaxation is not None:
                stress = relaxed.get_stress()
                if (stress.shape != (6,) or np.iscomplexobj(stress)
                        or not np.isfinite(stress).all()):
                    raise RuntimeError("Slab calculator must return six finite real stress components")
                max_stress = float(config.cell_height * np.abs(stress[[0, 1, 5]]).max())
                if not np.isfinite(max_stress):
                    raise RuntimeError("Slab calculator returned a nonfinite in-plane stress")
                stress_converged = max_stress < cell_relaxation.stress_max
            if max_force < fmax and stress_converged:
                relaxed.info.update(slab_quench_steps=steps, slab_quench_fmax=max_force)
                if cell_relaxation is not None:
                    relaxed.info["slab_quench_stress_max"] = max_stress
                return relaxed
            if steps < max_steps:
                optimizer.step()

    if cell_relaxation is not None:
        raise SlabQuenchNotConverged(
            f"Slab quench did not converge after {max_steps} steps: "
            f"max force {max_force:.6g} eV/Angstrom (fmax {fmax:.6g}); "
            f"max in-plane stress {max_stress:.6g} eV/Angstrom² "
            f"(stress_max {cell_relaxation.stress_max:.6g})")
    raise SlabQuenchNotConverged(
        f"Slab quench did not converge after {max_steps} steps: "
        f"max force {max_force:.6g} eV/Angstrom >= fmax {fmax:.6g}")
