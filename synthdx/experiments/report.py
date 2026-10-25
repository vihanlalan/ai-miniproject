"""Generate figures and an auto-written markdown report from ``results/``.

Every number in the report is read from the CSV files produced by the
experiment runner; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from synthdx.config import load_config, resolve_path  # noqa: E402
from synthdx.experiments.arms import arm_label  # noqa: E402
from synthdx.ml.models import DISPLAY_NAMES  # noqa: E402
from synthdx.palette import ARM_COLORS, ARM_ORDER, INK, MODEL_ORDER, ordered  # noqa: E402
from synthdx.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

HEADLINE = ["recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc", "accuracy"]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
    ax.tick_params(colors=INK["secondary"], labelsize=9)
    ax.yaxis.grid(True, color=INK["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.patch.set_facecolor(INK["surface"])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_metric_bars(metrics: pd.DataFrame, metric: str, path: Path) -> Path:
    """Grouped bars: model on x, one bar per arm, error bars = std over seeds."""
    models = ordered(metrics["model"].unique(), MODEL_ORDER)
    arms = ordered(metrics["arm"].unique(), ARM_ORDER)
    agg = metrics.groupby(["model", "arm"])[metric].agg(["mean", "std"]).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / len(arms)
    x = np.arange(len(models))
    for i, arm in enumerate(arms):
        means = [agg.loc[(m, arm), "mean"] if (m, arm) in agg.index else np.nan for m in models]
        stds = [agg.loc[(m, arm), "std"] if (m, arm) in agg.index else 0 for m in models]
        ax.bar(x + (i - (len(arms) - 1) / 2) * width, means, width * 0.92, yerr=stds,
               color=ARM_COLORS.get(arm), label=arm_label(arm),
               error_kw={"ecolor": INK["secondary"], "elinewidth": 1, "capsize": 2})
    ax.set_xticks(x, [DISPLAY_NAMES.get(m, m).replace(" (", "\n(") for m in models], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " "), color=INK["secondary"])
    ax.set_title(f"{metric.replace('_', ' ').title()} on the real test set "
                 "(mean ± std over seeds)", color=INK["primary"], fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return _save(fig, path)


def plot_roc_grid(preds: pd.DataFrame, path: Path) -> Path:
    """One panel per model; ROC curve per arm with predictions pooled over seeds."""
    models = ordered(preds["model"].unique(), MODEL_ORDER)
    cols = 3
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8 * rows), squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        sub = preds[preds.model == model]
        for arm in ordered(sub["arm"].unique(), ARM_ORDER):
            s = sub[sub.arm == arm]
            fpr, tpr, _ = roc_curve(s["y_true"], s["score"])
            ax.plot(fpr, tpr, color=ARM_COLORS.get(arm), linewidth=2, label=arm)
        ax.plot([0, 1], [0, 1], color=INK["axis"], linestyle="--", linewidth=1)
        ax.set_title(DISPLAY_NAMES.get(model, model), fontsize=9, color=INK["primary"], loc="left")
        ax.set_xlabel("false positive rate", fontsize=8, color=INK["secondary"])
        ax.set_ylabel("true positive rate", fontsize=8, color=INK["secondary"])
        _style(ax)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    return _save(fig, path)


def plot_losses(losses: pd.DataFrame, path: Path) -> Path:
    """Generator / discriminator loss curves for each GAN (first seed)."""
    gens = sorted(losses["generator"].unique())
    first = losses["seed"].min()
    fig, axes = plt.subplots(1, len(gens), figsize=(5.5 * len(gens), 3.6), squeeze=False)
    for ax, g in zip(axes.ravel(), gens):
        s = losses[(losses.generator == g) & (losses.seed == first)]
        ax.plot(s["epoch"], s["generator_loss"], color=ARM_COLORS["A_real"], linewidth=2,
                label="generator")
        ax.plot(s["epoch"], s["discriminator_loss"], color=ARM_COLORS["B_ctgan"], linewidth=2,
                label="discriminator")
        ax.set_title(f"{g.upper()} training loss (seed {first})", fontsize=10,
                     color=INK["primary"], loc="left")
        ax.set_xlabel("epoch", fontsize=8, color=INK["secondary"])
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_fairness(fair: pd.DataFrame, path: Path) -> Path:
    """Equal-opportunity difference per arm and model, one panel per attribute."""
    attrs = sorted(fair["attribute"].unique())
    fig, axes = plt.subplots(1, len(attrs), figsize=(6.5 * len(attrs), 4), squeeze=False)
    for ax, attr in zip(axes.ravel(), attrs):
        sub = fair[fair.attribute == attr]
        plot_df = sub.rename(columns={"equal_opportunity_diff": "v"})
        models = ordered(plot_df["model"].unique(), MODEL_ORDER)
        arms = ordered(plot_df["arm"].unique(), ARM_ORDER)
        agg = plot_df.groupby(["model", "arm"])["v"].mean()
        width = 0.8 / len(arms)
        x = np.arange(len(models))
        for i, arm in enumerate(arms):
            vals = [agg.get((m, arm), np.nan) for m in models]
            ax.bar(x + (i - (len(arms) - 1) / 2) * width, vals, width * 0.92,
                   color=ARM_COLORS.get(arm), label=arm)
        ax.set_xticks(x, [m.replace("_", "\n") for m in models], fontsize=7)
        ax.set_title(f"Equal opportunity difference by {attr} (lower = fairer)", fontsize=10,
                     color=INK["primary"], loc="left")
        _style(ax)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    return _save(fig, path)


def _fmt(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.3f} ± {0 if pd.isna(std) else std:.3f}"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _findings(metrics: pd.DataFrame, sig: pd.DataFrame) -> list[str]:
    """Plain statements computed from the results (no hand-written numbers)."""
    out = []
    agg = metrics.groupby(["model", "arm"])["recall"].mean()
    for model in ordered(metrics["model"].unique(), MODEL_ORDER):
        if (model, "A_real") not in agg.index:
            continue
        base = agg[(model, "A_real")]
        arms = agg[model]
        best_arm = arms.idxmax()
        line = (f"**{DISPLAY_NAMES.get(model, model)}**: minority recall {base:.3f} on arm A; "
                f"best arm {best_arm} with {arms.max():.3f}")
        if best_arm != "A_real" and len(sig):
            row = sig[(sig.model == model) & (sig.arm == best_arm) & (sig.metric == "recall")]
            if len(row):
                p = row["t_pvalue"].iloc[0]
                line += f" (paired t-test vs A: p = {p:.3g}{', significant' if p < 0.05 else ', not significant'} at 0.05)"
        out.append(line + ".")
    gan_arms = [a for a in metrics["arm"].unique() if a.startswith("B_")]
    if gan_arms and "C_smote" in metrics["arm"].unique():
        rec = metrics.groupby("arm")["recall"].mean()
        best_gan = max(gan_arms, key=lambda a: rec[a])
        verdict = "higher" if rec[best_gan] > rec["C_smote"] else "not higher"
        out.append(f"Averaged over all models, the best GAN-augmented arm ({best_gan}, recall "
                   f"{rec[best_gan]:.3f}) is {verdict} than SMOTE ({rec['C_smote']:.3f}).")
    return out


def generate_report(results_dir: str | Path) -> Path:
    """Create figures under ``results/figures`` and write ``results/report.md``."""
    rd = resolve_path(results_dir)
    figs = rd / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(rd / "metrics.csv")
    summary = pd.read_csv(rd / "metrics_summary.csv")
    sig_path = rd / "significance.csv"
    sig = pd.read_csv(sig_path) if sig_path.exists() and sig_path.stat().st_size > 1 else pd.DataFrame()
    meta = json.loads((rd / "run_metadata.json").read_text())

    plot_metric_bars(metrics, "recall", figs / "recall_by_arm.png")
    plot_metric_bars(metrics, "f1", figs / "f1_by_arm.png")
    plot_metric_bars(metrics, "balanced_accuracy", figs / "balanced_accuracy_by_arm.png")
    plot_roc_grid(pd.read_csv(rd / "predictions.csv"), figs / "roc_curves.png")
    losses = pd.read_csv(rd / "losses.csv") if (rd / "losses.csv").exists() else pd.DataFrame()
    if len(losses):
        plot_losses(losses, figs / "gan_losses.png")
    fair = pd.read_csv(rd / "fairness_summary.csv")
    plot_fairness(fair, figs / "fairness_eod.png")

    seeds = meta["config"]["experiments"]["seeds"]
    lines = [
        "# SynthDx results report",
        "",
        "> Auto-generated by `synthdx.experiments.report` from the CSV files in this folder. "
        "Academic demonstration only - not for clinical use.",
        "",
        f"- Run finished: {meta['timestamp']} (runtime {meta['runtime_seconds']} s, "
        f"Python {meta['python']})",
        f"- Dataset: UCI Heart Disease (Cleveland), {meta['n_records']} records; "
        f"training positives downsampled to {meta['config']['data']['positive_fraction']:.0%}",
        f"- Seeds: {seeds} (each seed = new split, new GANs, new models)",
        f"- Test set: real held-out records only "
        f"({meta['config']['data']['test_size']:.0%} of the data), identical for every arm within a seed",
        "",
        "## Key findings (computed)",
        "",
        *[f"- {f}" for f in _findings(metrics, sig)],
        "",
        "## Training arm sizes (mean over seeds)",
        "",
    ]
    sizes = pd.read_csv(rd / "arm_sizes.csv").groupby("arm")[["n", "n_pos", "n_synthetic"]].mean()
    sizes = sizes.reindex(ordered(sizes.index, ARM_ORDER)).round(1).reset_index()
    sizes["arm"] = sizes["arm"].map(arm_label)
    lines += [_md_table(sizes), "", "## Classification results (mean ± std over seeds)", ""]
    for model in ordered(summary["model"].unique(), MODEL_ORDER):
        s = summary[summary.model == model].copy()
        s = s.set_index("arm").reindex(ordered(s["arm"], ARM_ORDER)).reset_index()
        table = pd.DataFrame({"arm": s["arm"]})
        for m in HEADLINE:
            table[m] = [_fmt(a, b) for a, b in zip(s[f"{m}_mean"], s[f"{m}_std"])]
        lines += [f"### {DISPLAY_NAMES.get(model, model)}", "", _md_table(table), ""]
    lines += ["![Recall](figures/recall_by_arm.png)", "", "![F1](figures/f1_by_arm.png)", "",
              "![ROC](figures/roc_curves.png)", ""]

    if len(sig):
        rs = sig[sig.metric == "recall"].copy()
        rs = rs.assign(mean_diff=rs.mean_diff.round(3), t_pvalue=rs.t_pvalue.round(4),
                       wilcoxon_pvalue=rs.wilcoxon_pvalue.round(4))
        lines += ["## Paired significance tests vs arm A (minority recall)", "",
                  "Paired over seeds. With 5 seeds the Wilcoxon test cannot go below p = 0.0625, "
                  "so the paired t-test is primary. No multiple-comparison correction is applied.",
                  "", _md_table(rs[["model", "arm", "mean_diff", "t_pvalue", "wilcoxon_pvalue"]]), ""]

    gq_path = rd / "generator_quality_agg.csv"
    if gq_path.exists():
        gq = pd.read_csv(gq_path)
        keys = ["mean_ks_stat", "mean_js_distance", "corr_mean_abs_diff", "detectability_auc",
                "dcr_synth_median", "dcr_holdout_median", "dcr_ratio", "exact_duplicates",
                "mia_auc", "train_seconds"]
        t = pd.DataFrame({"generator": gq["generator"]})
        for k in keys:
            if f"{k}_mean" in gq:
                t[k] = [_fmt(a, b) for a, b in zip(gq[f"{k}_mean"], gq[f"{k}_std"])]
        lines += ["## Synthetic data fidelity and privacy (mean ± std over seeds)", "",
                  "Fidelity compares a synthetic sample (drawn at the training class prior) with the "
                  "real training data. Detectability AUC: 0.5 = indistinguishable. DCR ratio = median "
                  "synthetic-to-train distance / median holdout-to-train distance (< 1 hints at "
                  "memorisation). MIA AUC: 0.5 = membership not inferable.", "",
                  _md_table(t.T.reset_index().rename(columns=lambda c: str(c))
                            .rename(columns={"index": "metric"})), ""]
        if len(losses):
            lines += ["![GAN losses](figures/gan_losses.png)", ""]

    fa_path = rd / "fairness_summary_agg.csv"
    if fa_path.exists():
        fa = pd.read_csv(fa_path)
        fa = fa.assign(DPD=[_fmt(a, b) for a, b in zip(fa.demographic_parity_diff_mean,
                                                       fa.demographic_parity_diff_std)],
                       EOD=[_fmt(a, b) for a, b in zip(fa.equal_opportunity_diff_mean,
                                                       fa.equal_opportunity_diff_std)])
        lines += ["## Fairness (mean ± std over seeds)", "",
                  "DPD = demographic parity difference, EOD = equal opportunity difference "
                  "(both: max - min across groups; 0 is best). Small groups make these noisy.", "",
                  _md_table(fa[["attribute", "model", "arm", "DPD", "EOD"]]
                            .sort_values(["attribute", "model", "arm"])), "",
                  "![Fairness](figures/fairness_eod.png)", ""]

    ep_path = rd / "expert_params.csv"
    if ep_path.exists():
        ep = pd.read_csv(ep_path)
        th = ep[ep.param == "threshold"].groupby(["model", "arm"])["value"].agg(["mean", "std"])
        th = th.reset_index().round(3)
        lines += ["## Expert-system calibration per arm", "",
                  "The hand-written rules never change; `expert_calibrated` only learns a CF "
                  "decision threshold on each training arm, `expert_tuned` also re-estimates the "
                  "CFs of the diagnosis rules (see docs/methodology.md).", "", _md_table(th), ""]

    report = rd / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report)
    return report


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    generate_report(load_config(args.config)["paths"]["results_dir"])


if __name__ == "__main__":  # pragma: no cover
    main()
