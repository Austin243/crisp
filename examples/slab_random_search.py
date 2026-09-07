"""Small Cu4/EMT demonstration of the opt-in 2D random-search driver.

Run from the repository with CRISP, PyXtal and torch_fplib importable.
This checks search mechanics; it does not establish a stable copper monolayer.
"""

import argparse
from pathlib import Path

from ase.calculators.emt import EMT

from crisp.slab import SlabConfig
from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_search import SlabRandomSearch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8, help="additional trials, including rejections")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--checkpoint", type=Path, default=Path("slab-search.json"))
    parser.add_argument("--resume", action="store_true", help="load the checkpoint before continuing")
    args = parser.parse_args()
    if args.trials < 0 or args.seed < 0:
        parser.error("trials and seed must be nonnegative")
    if args.checkpoint.exists() and not args.resume:
        parser.error("checkpoint exists; use --resume or choose a new --checkpoint path")

    config = SlabConfig(
        area_per_atom_range=(4.5, 6.0), initial_thickness=1.0,
        max_thickness=3.0, cell_height=18.0, min_vacuum=12.0,
    )
    archive = SlabArchive(
        SlabFingerprintCalculator(config, cutoff=3.2, natx=32), {"Cu": 4},
        min_dist_ang=1.5,
    )
    search = SlabRandomSearch(
        archive, EMT, calculator_id="ASE-EMT-default", seed=args.seed,
        layer_groups=[1, 2, 80], max_generation_attempts=6, fmax=0.05, max_steps=200,
    )
    if args.resume:
        search.load(args.checkpoint)
    search.run(args.trials, checkpoint=args.checkpoint)
    print(f"Completed trials: {search.completed_trials}; outcomes: {search.counts}")
    print(f"Distinct validated slabs: {len(archive.entries)}")
    for rank, entry in enumerate(archive.get_best(5), 1):
        print(f"{rank}: {entry.energy:.8f} eV/atom (trial {entry.metadata['search_trial']})")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
