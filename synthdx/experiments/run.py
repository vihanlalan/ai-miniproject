"""CLI: run all experiment arms over all seeds, then write the report.

    python -m synthdx.experiments.run                 # full run (config.yaml)
    python -m synthdx.experiments.run --quick         # 2 seeds, few epochs
"""

from __future__ import annotations

import argparse

from synthdx.config import load_config
from synthdx.experiments.report import generate_report
from synthdx.experiments.runner import run_experiments

QUICK_OVERRIDES = {
    "experiments": {"seeds": [0, 1]},
    "gan": {"ctgan": {"epochs": 30}, "cgan": {"epochs": 100}},
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--seeds", type=int, nargs="+", help="override seeds")
    args = parser.parse_args(argv)
    overrides = dict(QUICK_OVERRIDES) if args.quick else {}
    if args.seeds:
        overrides["experiments"] = {"seeds": args.seeds}
    cfg = load_config(args.config, overrides)
    run_experiments(cfg)
    generate_report(cfg["paths"]["results_dir"])


if __name__ == "__main__":  # pragma: no cover
    main()
