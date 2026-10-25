"""Train the demo generators used by the UI and cache their artefacts.

Fits CTGAN and the custom cGAN on the imbalanced training split for the
config's master seed and writes:
    models/<gen>.pkl                    fitted generator
    results/demo/losses_<gen>.csv       per-epoch loss curve
    results/demo/synthetic_<gen>.csv    balanced synthetic sample
    results/demo/column_fidelity_<gen>.csv
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from synthdx.config import load_config, resolve_path
from synthdx.data import load_from_config, split_from_config
from synthdx.evaluation.fidelity import column_fidelity, fidelity_summary
from synthdx.evaluation.privacy import privacy_summary
from synthdx.gan import build_generator
from synthdx.utils import get_logger

logger = get_logger(__name__)


def train_demo_models(cfg: dict, generators: list[str] | None = None,
                      epochs: dict[str, int] | None = None, n_samples: int = 300) -> dict:
    """Train and save demo generators; returns a summary dict."""
    seed = cfg["seed"]
    df = load_from_config(cfg)
    split = split_from_config(df, cfg, seed)
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    demo_dir = resolve_path(cfg["paths"]["results_dir"]) / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in generators or cfg["experiments"]["generators"]:
        t0 = time.time()
        gen = build_generator(name, cfg, seed, (epochs or {}).get(name)).fit(split.train)
        secs = time.time() - t0
        gen.save(models_dir / f"{name}.pkl")
        gen.loss_history.to_csv(demo_dir / f"losses_{name}.csv", index=False)
        half = n_samples // 2
        synth = pd.concat([gen.generate(half, 0), gen.generate(n_samples - half, 1)],
                          ignore_index=True)
        synth.to_csv(demo_dir / f"synthetic_{name}.csv", index=False)
        sample = gen.generate(len(split.train))
        column_fidelity(split.train, sample).to_csv(
            demo_dir / f"column_fidelity_{name}.csv", index=False)
        summary[name] = {"train_seconds": round(secs, 1),
                         **fidelity_summary(split.train, sample, seed),
                         **privacy_summary(sample, split.train, split.test)}
        logger.info("%s trained in %.1fs", name, secs)
    (demo_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--quick", action="store_true", help="few epochs (smoke test)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    epochs = cfg["ui"]["quick_epochs"] if args.quick else None
    summary = train_demo_models(cfg, epochs=epochs)
    logger.info("Demo summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
