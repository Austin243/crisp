"""Compare random and GP-screened Cu4 searches at equal physical-quench budgets.

This records search mechanics and cost; it does not establish material stability
or promise a GP improvement. Requires PyXtal and torch_fplib to be importable.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

import ase
from ase.calculators.emt import EMT
import numpy as np
import pyxtal

from crisp.slab import SlabConfig
from crisp.slab_archive import SlabArchive
from crisp.slab_fingerprint import SlabFingerprintCalculator
from crisp.slab_screening import SlabGPScreening
from crisp.slab_search import SlabRandomSearch


CONFIG = SlabConfig((4.5, 6.0), 1.0, 3.0, 18.0, 12.0)


def run_arm(seed, screening, quenches, max_trials):
    counts = dict(quench_starts=0, calculator_evaluations=0,
                  generator_requests=0, fingerprint_calls=0)

    class CountedEMT(EMT):
        def calculate(self, *args, **kwargs):
            counts["calculator_evaluations"] += 1
            return super().calculate(*args, **kwargs)

    def factory():
        counts["quench_starts"] += 1
        return CountedEMT()

    class CountedFingerprints(SlabFingerprintCalculator):
        def get_fingerprints(self, atoms):
            counts["fingerprint_calls"] += 1
            return super().get_fingerprints(atoms)

    class CountedSearch(SlabRandomSearch):
        def _generate(self, seed):
            counts["generator_requests"] += 1
            return super()._generate(seed)

    archive = SlabArchive(CountedFingerprints(CONFIG, cutoff=3.2, natx=64),
                          {"Cu": 4}, min_dist_ang=1.5)
    search = CountedSearch(archive, factory, calculator_id="ASE-EMT-default", seed=seed,
                           layer_groups=[1, 2, 80], max_generation_attempts=6,
                           fmax=0.05, max_steps=200, screening=screening)
    while counts["quench_starts"] < quenches and search.completed_trials < max_trials:
        search.run(1)
    if counts["quench_starts"] != quenches:
        raise RuntimeError("Trial cap exhausted before equal quench budget; compare only complete arms")
    best = archive.get_best(1)
    return dict(seed=seed, mode="gp" if screening else "random", costs=counts,
                best_energy_per_atom=best[0].energy if best else None,
                distinct_slabs=len(archive.entries), outcomes=search.outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 31])
    parser.add_argument("--quenches", type=int, default=12)
    parser.add_argument("--max-trials", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("slab-gp-benchmark.json"))
    args = parser.parse_args()
    if args.quenches < 1 or args.max_trials < args.quenches or min(args.seeds) < 0:
        parser.error("require nonnegative seeds and 1 <= quenches <= max-trials")
    if args.output.exists():
        parser.error("output exists; choose a new --output path")
    rows = []
    for seed in args.seeds:
        for screening in (None, SlabGPScreening()):
            row = run_arm(seed, screening, args.quenches, args.max_trials)
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "outcomes"}), flush=True)
    report = dict(python=platform.python_version(), ase=ase.__version__, numpy=np.__version__,
                  pyxtal=pyxtal.__version__, slab=asdict(CONFIG), screening=asdict(SlabGPScreening()),
                  quench_budget=args.quenches, max_trials=args.max_trials, results=rows)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
