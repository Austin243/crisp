"""Small native-VASP MoS2 search example; not a validated material recipe.

Requires licensed VASP >=6.5.0, matching local POTCAR datasets and torch_fplib.
Run on compute resources, e.g. inside an existing allocation. Converge the
physical inputs for your chemistry before interpreting the candidate energies.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crisp.slab import SlabConfig
from crisp.slab_crisp_search import SlabCRISPSearch
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_layer_archive import LayeredSlabArchive
from crisp.slab_layers import InitialLayer, LayeredSlabSpec
from crisp.slab_vasp import NativeVASPConfig, SlabVASPRelaxer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vasp-version", required=True, help="Exact supported build, e.g. 6.5.1")
    parser.add_argument("--mo-potcar", required=True, type=Path)
    parser.add_argument("--s-potcar", required=True, type=Path)
    parser.add_argument("--encut", required=True, type=float, help="Chosen cutoff in eV")
    parser.add_argument("--command", nargs="+", default=["vasp_std"], help="Launcher argv; no shell syntax")
    parser.add_argument("--work-dir", type=Path, default=Path("output/native-mos2"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--budget", type=int, default=8, help="Total accepted/rejected native candidate attempts")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    geometry = SlabConfig(area_per_atom_range=(2.5, 5.0), initial_thickness=0,
                          max_thickness=8, cell_height=24, min_vacuum=16)
    spec = LayeredSlabSpec(
        layers=(InitialLayer({"S": 2}), InitialLayer({"Mo": 2}), InitialLayer({"S": 2})),
        config=geometry, initial_layer_gaps=(1.5, 1.5), min_dist_ang=1.1,
        pair_min_distances={("Mo", "Mo"): 2.0, ("Mo", "S"): 1.6, ("S", "S"): 1.5})
    fingerprint = SlabFingerprintCalculator(geometry, cutoff=4.5, natx=96)
    archive = LayeredSlabArchive(fingerprint, spec)
    backend = SlabVASPRelaxer(
        spec, NativeVASPConfig(command=tuple(args.command), version=args.vasp_version,
                               potentials={"Mo": str(args.mo_potcar), "S": str(args.s_potcar)},
                               incar={"ENCUT": args.encut, "ISMEAR": 0, "SIGMA": 0.05},
                               kpoints=(6, 6, 1)), args.work_dir / "candidates")
    search = SlabCRISPSearch(archive, backend, seed=args.seed, n_random=4, n_mutants=2,
                             max_generations=args.generations, budget_relax=args.budget)
    checkpoint = args.work_dir / "search.json"
    if args.resume:
        search.load(checkpoint)
    search.run(checkpoint=checkpoint)
    archive.save(args.work_dir / "archive.json")
    print(f"Stopped: {search.stop_reason}; {search.n_relaxed} attempts; {len(archive.entries)} unique candidates")
    for entry in archive.get_best(5):
        print(f"{entry.metadata['candidate_id']}: {entry.energy:.8f} eV/atom")


if __name__ == "__main__":
    main()
