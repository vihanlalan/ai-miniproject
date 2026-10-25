"""The seven pages of the SynthDx Streamlit app."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

from synthdx.app.components import (
    get_cached_generator,
    get_config,
    get_dataset,
    get_engine,
    get_rules_df,
    get_split,
    missing_results_message,
    read_metadata,
    read_result,
    results_dir,
    risk_pill,
    stat_card,
    style_fig,
)
from synthdx.data.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    NUMERIC_COLUMNS,
    TARGET,
)
from synthdx.evaluation.fidelity import column_fidelity, correlation_matrices
from synthdx.expert.explain import explain_how, trace_to_dataframe
from synthdx.experiments.arms import arm_label
from synthdx.gan import build_generator
from synthdx.ml.models import DISPLAY_NAMES
from synthdx.palette import (
    ARM_COLORS,
    ARM_ORDER,
    CLASS_COLORS,
    DATA_COLORS,
    DIVERGING,
    MODEL_ORDER,
    STATUS,
    ordered,
)

CHOICES = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "No (<= 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}
NUMERIC_INPUTS = {  # min, max, step
    "age": (18, 100, 1), "trestbps": (80, 220, 1), "chol": (100, 600, 1),
    "thalach": (60, 220, 1), "oldpeak": (0.0, 7.0, 0.1),
}
DEFAULT_PATIENT = dict(age=58, sex=1, cp=4, trestbps=140, chol=250, fbs=0, restecg=0,
                       thalach=140, exang=1, oldpeak=1.4, slope=2, ca=1, thal=7)
CLASS_NAMES = {0: "No disease", 1: "Disease"}


def _model_name(m: str) -> str:
    return DISPLAY_NAMES.get(m, m)


# =================================================================== 1. Overview
def page_overview() -> None:
    st.title("SynthDx")
    st.markdown("#### Synthetic medical data generation + a diagnostic expert system")
    st.markdown(
        '<div class="sdx-callout"><b>Research question.</b> Does augmenting a real, imbalanced '
        "medical dataset with GAN-generated synthetic records improve the diagnostic accuracy "
        "(especially minority-class recall) of an expert system and ML baselines, without "
        "leaking privacy or worsening fairness?</div>", unsafe_allow_html=True)

    cfg = get_config()
    df = get_dataset()
    split = get_split()
    rules = get_rules_df()
    c1, c2, c3, c4 = st.columns(4)
    stat_card(c1, "Real records", f"{len(df)}", "UCI Heart Disease (Cleveland)")
    stat_card(c2, "Training positives", f"{split['train'][TARGET].mean():.0%}",
              f"downsampled from {df[TARGET].mean():.0%} to simulate imbalance")
    stat_card(c3, "Expert rules", f"{len(rules)}", "hand-written, MYCIN certainty factors")
    stat_card(c4, "Seeds per experiment", f"{len(cfg['experiments']['seeds'])}",
              "mean ± std reported everywhere")

    st.subheader("Pipeline")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
            color="#c3c2b7", fillcolor="#ffffff"];
      edge [color="#898781"];
      data [label="UCI Heart Disease\\n(303 real records)", fillcolor="#eef4fc"];
      split [label="Stratified split\\n+ imbalance simulation"];
      test [label="Held-out REAL test set\\n(never augmented)", fillcolor="#fff6e0"];
      gan [label="CTGAN  |  custom cGAN\\n(class-conditional)"];
      arms [label="Arms A-D\\nreal / +GAN / +SMOTE / synthetic-only"];
      ml [label="ML baselines\\nLR, RF, GB"];
      es [label="Expert system\\nforward chaining + CFs"];
      ev [label="Evaluation\\nmetrics, fidelity,\\nprivacy, fairness", fillcolor="#eef4fc"];
      data -> split; split -> test; split -> gan; split -> arms; gan -> arms;
      arms -> ml; arms -> es [label=" calibrate"]; ml -> ev; es -> ev; test -> ev;
    }""", use_container_width=True)

    st.subheader("What the app shows")
    st.markdown(
        "- **Data Explorer**: the real data and the imbalance we inject.\n"
        "- **GAN Lab**: train or load CTGAN / cGAN, generate records, compare real vs synthetic.\n"
        "- **Expert System Diagnosis**: enter a patient, see the rule trace, CF and what-if analysis.\n"
        "- **Experiments Dashboard**: arms A-D compared on the same real test set over 5 seeds.\n"
        "- **Fairness & Privacy**: group gaps, distance-to-closest-record, membership inference.\n"
        "- **Rule Browser**: every rule with its certainty factor and rationale.")

    metrics = read_result("metrics")
    if metrics is not None:
        st.subheader("Headline result (computed from the saved results)")
        rec = metrics.groupby(["model", "arm"])["recall"].mean().unstack()
        rec = rec.reindex(index=ordered(rec.index, MODEL_ORDER), columns=ordered(rec.columns, ARM_ORDER))
        rec.index = [_model_name(m) for m in rec.index]
        st.markdown("Mean minority-class **recall** on the real test set:")
        st.dataframe(rec.style.format("{:.3f}").background_gradient(cmap="Blues", axis=1),
                     use_container_width=True)


# =================================================================== 2. Data Explorer
def page_data() -> None:
    st.title("Data Explorer")
    df = get_dataset()
    split = get_split()
    st.markdown(
        "UCI Heart Disease (Cleveland): 303 patients, 13 clinical features, target = "
        "angiographic disease (`num > 0`). Public, de-identified research data; **no private "
        "patient data is used**.")

    c1, c2, c3 = st.columns(3)
    stat_card(c1, "Full dataset", f"{len(df)} rows", f"{df[TARGET].mean():.1%} positive")
    stat_card(c2, "Imbalanced training set", f"{len(split['train'])} rows",
              f"{split['train'][TARGET].mean():.1%} positive (minority)")
    stat_card(c3, "Real test set", f"{len(split['test'])} rows",
              f"{split['test'][TARGET].mean():.1%} positive, never augmented")

    st.subheader("Class balance")
    rows = []
    for name, frame in (("Full dataset", df), ("Training pool", split["train_pool"]),
                        ("Imbalanced train", split["train"]), ("Test", split["test"])):
        for cls in (0, 1):
            rows.append({"set": name, "class": CLASS_NAMES[cls],
                         "count": int((frame[TARGET] == cls).sum())})
    fig = px.bar(pd.DataFrame(rows), x="set", y="count", color="class", barmode="group",
                 text="count", color_discrete_map={CLASS_NAMES[k]: v for k, v in CLASS_COLORS.items()})
    fig.update_traces(textposition="outside", marker_line_width=0)
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    st.subheader("Feature distributions")
    col = st.selectbox("Feature", FEATURE_COLUMNS, format_func=lambda c: FEATURE_LABELS[c],
                       key="data_feature")
    plot_df = df.assign(cls=df[TARGET].map(CLASS_NAMES))
    if col in NUMERIC_COLUMNS:
        fig = px.histogram(plot_df, x=col, color="cls", barmode="overlay", nbins=25, opacity=0.7,
                           color_discrete_map={CLASS_NAMES[k]: v for k, v in CLASS_COLORS.items()},
                           labels={col: FEATURE_LABELS[col], "cls": "class"})
    else:
        counts = plot_df.groupby([col, "cls"]).size().reset_index(name="count")
        counts[col] = counts[col].map(CHOICES.get(col, {})).fillna(counts[col].astype(str))
        fig = px.bar(counts, x=col, y="count", color="cls", barmode="group",
                     color_discrete_map={CLASS_NAMES[k]: v for k, v in CLASS_COLORS.items()},
                     labels={col: FEATURE_LABELS[col], "cls": "class"})
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    with st.expander("Summary statistics"):
        st.dataframe(df.describe().T.round(2), use_container_width=True)
    with st.expander("Raw records (first 50)"):
        st.dataframe(df.head(50), use_container_width=True)


# =================================================================== 3. GAN Lab
def _loss_figure(losses: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=losses["epoch"], y=losses["generator_loss"], name="Generator",
                    line=dict(color=DATA_COLORS["real"], width=2))
    fig.add_scatter(x=losses["epoch"], y=losses["discriminator_loss"], name="Discriminator",
                    line=dict(color=DATA_COLORS["synthetic"], width=2))
    fig.update_layout(xaxis_title="epoch", yaxis_title="loss", hovermode="x unified")
    return style_fig(fig, 320, title)


def page_gan_lab() -> None:
    st.title("GAN Lab")
    st.markdown("Generate synthetic patient records with **CTGAN** (library) or a **custom "
                "conditional GAN** written from scratch in PyTorch. Cached models load instantly; "
                "quick retrain uses a handful of epochs.")
    cfg = get_config()
    split = get_split()
    c1, c2 = st.columns([1, 1])
    model = c1.radio("Model", ["ctgan", "cgan"], horizontal=True, key="gan_model",
                     format_func=lambda m: {"ctgan": "CTGAN", "cgan": "Custom cGAN"}[m])
    mode = c2.radio("Source", ["Cached (pre-trained)", "Quick retrain"], horizontal=True,
                    key="gan_mode")

    store_key = f"gen_{model}"
    if mode == "Quick retrain":
        default = cfg["ui"]["quick_epochs"][model]
        epochs = st.slider("Epochs", 5, 300, default, step=5, key="gan_epochs")
        if st.button("Train now", type="primary", key="gan_train"):
            try:
                with st.spinner(f"Training {model.upper()} for {epochs} epochs..."):
                    t0 = time.time()
                    gen = build_generator(model, cfg, cfg["seed"], epochs).fit(split["train"])
                st.session_state[store_key] = gen
                st.success(f"Trained in {time.time() - t0:.1f}s on {len(split['train'])} "
                           "imbalanced real training records.")
            except Exception as exc:  # friendly message instead of a stack trace
                st.error(f"Training failed: {exc}")
        gen = st.session_state.get(store_key)
    else:
        gen = get_cached_generator(model)
        if gen is None:
            st.warning("No cached model found. Run `make train`, or switch to *Quick retrain*.")

    if gen is None:
        st.info("Train or load a model to continue.")
        return

    if len(gen.loss_history):
        st.plotly_chart(_loss_figure(gen.loss_history, f"{model.upper()} training losses"),
                        use_container_width=True)
        with st.expander("How to read the loss curves"):
            st.markdown(
                "GAN losses do not decrease monotonically like a supervised loss: generator and "
                "discriminator compete. A healthy run shows both losses settling into a band. A "
                "discriminator loss collapsing to ~0 means it wins easily (poor samples); wild "
                "oscillation or a generator stuck on few patterns hints at **mode collapse**, "
                "which is checked below via distributions and detectability.")

    st.subheader("Generate records")
    g1, g2, g3 = st.columns([1, 1, 1])
    n = g1.number_input("Number of records", 10, 5000, 200, step=10, key="gan_n")
    cond = g2.selectbox("Class", ["Any (training prior)", "Disease (1)", "No disease (0)"],
                        key="gan_cond")
    condition = {"Any (training prior)": None, "Disease (1)": 1, "No disease (0)": 0}[cond]
    if g3.button("Generate", type="primary", key="gan_generate"):
        try:
            st.session_state["synthetic"] = gen.generate(int(n), condition)
            st.session_state["synthetic_model"] = model
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    synth = st.session_state.get("synthetic")
    if synth is None or st.session_state.get("synthetic_model") != model:
        st.caption("Press **Generate** to sample records.")
        return
    st.dataframe(synth.head(100), use_container_width=True, height=260)
    st.download_button("Download CSV", synth.to_csv(index=False).encode(),
                       f"synthetic_{model}.csv", "text/csv", key="gan_download")

    st.subheader("Real vs synthetic")
    real = split["train"] if condition is None else split["train"][split["train"][TARGET] == condition]
    col = st.selectbox("Feature", FEATURE_COLUMNS, format_func=lambda c: FEATURE_LABELS[c],
                       key="gan_feature")
    both = pd.concat([real.assign(source="real"), synth.assign(source="synthetic")])
    if col in NUMERIC_COLUMNS:
        fig = px.histogram(both, x=col, color="source", barmode="overlay", histnorm="probability",
                           nbins=25, opacity=0.65, color_discrete_map=DATA_COLORS)
    else:
        freq = both.groupby("source")[col].value_counts(normalize=True).rename("share").reset_index()
        freq[col] = freq[col].astype(str)
        fig = px.bar(freq, x=col, y="share", color="source", barmode="group",
                     color_discrete_map=DATA_COLORS)
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    if len(real) > 5 and len(synth) > 5:
        fid = column_fidelity(real, synth)
        st.markdown("**Per-column fidelity** (KS statistic for numerics, Jensen-Shannon distance "
                    "for all; 0 = identical)")
        st.dataframe(fid.round(3), use_container_width=True, hide_index=True)
        cr, cs = correlation_matrices(real, synth)
        h1, h2 = st.columns(2)
        for holder, mat, title in ((h1, cr, "Real correlations"), (h2, cs, "Synthetic correlations")):
            fig = px.imshow(mat.round(2), zmin=-1, zmax=1, color_continuous_scale=DIVERGING,
                            aspect="auto")
            holder.plotly_chart(style_fig(fig, 420, title), use_container_width=True)


# =================================================================== 4. Diagnosis
def _set_patient(record: dict, source: str) -> None:
    for col in FEATURE_COLUMNS:
        v = record[col]
        st.session_state[f"f_{col}"] = float(v) if col == "oldpeak" else int(v)
    st.session_state["patient_source"] = source


def _load_random(kind: str) -> None:
    rng = np.random.default_rng()
    if kind == "real":
        frame = get_split()["test"]
    else:
        frame = st.session_state.get("synthetic")
        if frame is None:
            path = results_dir() / "demo" / "synthetic_ctgan.csv"
            frame = pd.read_csv(path) if path.exists() else None
        if frame is None:
            st.session_state["patient_source"] = "No synthetic data available yet - generate some in GAN Lab."
            return
    row = frame.iloc[int(rng.integers(len(frame)))].to_dict()
    _set_patient(row, f"Random {kind} record (true label: {CLASS_NAMES[int(row[TARGET])]})")
    st.session_state.pop("diagnosed", None)


def _gauge(cf: float, threshold: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=cf, number={"valueformat": "+.2f"},
        title={"text": "Certainty factor for heart disease"},
        gauge={"axis": {"range": [-1, 1]}, "bar": {"color": "#0b0b0b", "thickness": 0.25},
               "steps": [{"range": [-1, 0.2], "color": "#d7f0d7"},
                         {"range": [0.2, 0.6], "color": "#fbe2d6"},
                         {"range": [0.6, 1], "color": "#f5d0d0"}],
               "threshold": {"line": {"color": STATUS["critical"], "width": 3},
                             "value": threshold}}))
    return style_fig(fig, 260)


def page_diagnosis() -> None:
    st.title("Expert System Diagnosis")
    st.markdown("A forward-chaining rule engine with MYCIN certainty factors. Enter a patient or "
                "load one, then press **Diagnose**.")
    for col, val in DEFAULT_PATIENT.items():
        st.session_state.setdefault(f"f_{col}", float(val) if col == "oldpeak" else val)

    b1, b2, _ = st.columns([1, 1, 2])
    b1.button("Random real patient", on_click=_load_random, args=("real",), key="load_real")
    b2.button("Random synthetic patient", on_click=_load_random, args=("synthetic",),
              key="load_synth")
    if st.session_state.get("patient_source"):
        st.caption(st.session_state["patient_source"])

    with st.form("patient_form"):
        cols = st.columns(4)
        for i, col in enumerate(FEATURE_COLUMNS):
            holder = cols[i % 4]
            if col in NUMERIC_INPUTS:
                lo, hi, step = NUMERIC_INPUTS[col]
                holder.number_input(FEATURE_LABELS[col], lo, hi, step=step, key=f"f_{col}")
            else:
                opts = list(CHOICES[col])
                holder.selectbox(FEATURE_LABELS[col], opts, key=f"f_{col}",
                                 format_func=lambda v, c=col: CHOICES[c][v])
        submitted = st.form_submit_button("Diagnose", type="primary")
    if submitted:
        st.session_state["diagnosed"] = {c: st.session_state[f"f_{c}"] for c in FEATURE_COLUMNS}
        for col in ("age", "trestbps", "chol", "thalach", "oldpeak", "ca"):
            st.session_state[f"w_{col}"] = st.session_state["diagnosed"][col]

    patient = st.session_state.get("diagnosed")
    if not patient:
        return
    engine = get_engine()
    try:
        d = engine.diagnose(patient)
    except Exception as exc:
        st.error(f"Could not diagnose this record: {exc}")
        return

    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown(f"### {d.label}")
        st.markdown(risk_pill(d.risk_level), unsafe_allow_html=True)
        st.markdown(f"\nCF = **{d.cf:+.3f}** (positive if CF >= {d.threshold:+.2f}). "
                    f"{len(d.trace)} rules fired.")
        if d.recommendations:
            st.markdown("**Recommendations (demo logic):** " + ", ".join(
                r.replace("recommend_", "").replace("_", " ") for r in d.recommendations))
    r2.plotly_chart(_gauge(d.cf, d.threshold), use_container_width=True)

    t1, t2, t3, t4 = st.tabs(["How: rule trace", "Why not: near misses", "Working memory",
                              "What-if"])
    with t1:
        st.dataframe(trace_to_dataframe(d), use_container_width=True, hide_index=True)
        for line in explain_how(d, engine):
            st.markdown(f"- {line}")
    with t2:
        if not d.near_misses:
            st.write("No rule came close to firing.")
        for m in d.near_misses:
            with st.expander(f"{m.rule_id} - {m.rule_name}  ({m.satisfied}/{m.total} conditions held)"):
                st.markdown(f"Would have concluded **{m.conclusion}** with rule CF {m.rule_cf:+.2f}.")
                for f in m.failed:
                    st.markdown(f"- failed: `{f}`")
    with t3:
        wm = pd.DataFrame([{"fact": k, **v} for k, v in d.working_memory.items()])
        wm["value"] = wm["value"].astype(str)
        st.dataframe(wm, use_container_width=True, hide_index=True, height=420)
    with t4:
        _what_if(engine, patient, d.cf)


def _what_if(engine, patient: dict, base_cf: float) -> None:
    st.markdown("Change one or more values and see how the conclusion moves.")
    for col in ("age", "trestbps", "chol", "thalach", "oldpeak", "ca"):
        st.session_state.setdefault(f"w_{col}", patient[col])
    c = st.columns(3)
    mod = dict(patient)
    for i, col in enumerate(("age", "trestbps", "chol", "thalach", "oldpeak")):
        lo, hi, step = NUMERIC_INPUTS[col]
        mod[col] = c[i % 3].slider(FEATURE_LABELS[col], lo, hi, step=step, key=f"w_{col}")
    mod["ca"] = c[2].select_slider(FEATURE_LABELS["ca"], [0, 1, 2, 3], key="w_ca")
    d2 = engine.diagnose(mod)
    m1, m2, m3 = st.columns(3)
    m1.metric("What-if CF", f"{d2.cf:+.3f}", f"{d2.cf - base_cf:+.3f}")
    m2.metric("Diagnosis", d2.label.replace("Heart disease ", ""))
    m3.metric("Risk level", d2.risk_level.upper())

    var = st.selectbox("Sweep one variable", list(NUMERIC_INPUTS) + ["ca"], key="w_sweep",
                       format_func=lambda v: FEATURE_LABELS[v])
    lo, hi, _ = NUMERIC_INPUTS.get(var, (0, 3, 1))
    grid = np.linspace(lo, hi, 40) if var != "ca" else np.arange(4)
    cfs = [engine.diagnose({**mod, var: v}).cf for v in grid]
    fig = go.Figure()
    fig.add_scatter(x=grid, y=cfs, mode="lines", line=dict(color=ARM_COLORS["A_real"], width=2),
                    name="CF", hovertemplate=f"{var}=%{{x}}<br>CF=%{{y:+.3f}}<extra></extra>")
    fig.add_hline(y=engine.decision_threshold, line_dash="dash", line_color=STATUS["critical"],
                  annotation_text="decision threshold")
    fig.add_vline(x=mod[var], line_color="#898781")
    fig.update_layout(xaxis_title=FEATURE_LABELS[var], yaxis_title="CF(heart disease)",
                      yaxis_range=[-1.05, 1.05], showlegend=False)
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)


# =================================================================== 5. Experiments
def page_experiments() -> None:
    st.title("Experiments Dashboard")
    metrics = read_result("metrics")
    if metrics is None:
        missing_results_message()
        return
    meta = read_metadata() or {}
    seeds = sorted(metrics["seed"].unique())
    st.markdown(f"Arms A-D evaluated on the **same real test set** for each of {len(seeds)} seeds "
                f"({', '.join(map(str, seeds))}). Last run: {meta.get('timestamp', 'n/a')}.")
    with st.expander("What are the arms?"):
        for arm in ordered(metrics["arm"].unique(), ARM_ORDER):
            st.markdown(f"- **{arm}**: {arm_label(arm)}")
        sizes = read_result("arm_sizes")
        if sizes is not None:
            st.dataframe(sizes.groupby("arm")[["n", "n_pos", "n_synthetic"]].mean().round(1),
                         use_container_width=True)

    c1, c2 = st.columns([1, 2])
    metric = c1.selectbox("Metric", ["recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc",
                                     "precision", "specificity", "accuracy"], key="exp_metric")
    models = c2.multiselect("Models", ordered(metrics["model"].unique(), MODEL_ORDER),
                            default=ordered(metrics["model"].unique(), MODEL_ORDER),
                            format_func=_model_name, key="exp_models")
    sub = metrics[metrics.model.isin(models)]
    if sub.empty:
        st.warning("Select at least one model.")
        return
    agg = sub.groupby(["model", "arm"])[metric].agg(["mean", "std"]).reset_index()
    agg["model_name"] = agg["model"].map(_model_name)
    fig = px.bar(agg, x="model_name", y="mean", color="arm", error_y="std", barmode="group",
                 color_discrete_map=ARM_COLORS,
                 category_orders={"arm": ARM_ORDER,
                                  "model_name": [_model_name(m) for m in ordered(models, MODEL_ORDER)]},
                 labels={"mean": metric, "model_name": ""},
                 hover_data={"mean": ":.3f", "std": ":.3f"})
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(style_fig(fig, 420, f"{metric} by model and arm (mean ± std over seeds)"),
                    use_container_width=True)

    st.subheader("Seed variance")
    m_one = st.selectbox("Model", ordered(models, MODEL_ORDER), format_func=_model_name,
                         key="exp_seed_model")
    one = metrics[metrics.model == m_one]
    fig = px.strip(one, x="arm", y=metric, color="arm", color_discrete_map=ARM_COLORS,
                   category_orders={"arm": ARM_ORDER}, hover_data=["seed"])
    fig.update_traces(marker_size=10)
    st.plotly_chart(style_fig(fig, 340, f"{_model_name(m_one)}: one dot per seed"),
                    use_container_width=True)

    preds = read_result("predictions")
    if preds is not None:
        st.subheader("ROC curves (predictions pooled over seeds)")
        p = preds[preds.model == m_one]
        fig = go.Figure()
        for arm in ordered(p["arm"].unique(), ARM_ORDER):
            s = p[p.arm == arm]
            fpr, tpr, _ = roc_curve(s["y_true"], s["score"])
            auc = roc_auc_score(s["y_true"], s["score"])
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{arm} (AUC {auc:.3f})",
                            line=dict(color=ARM_COLORS.get(arm), width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                        line=dict(color="#c3c2b7", dash="dash"))
        fig.update_layout(xaxis_title="false positive rate", yaxis_title="true positive rate")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.subheader("Summary table (mean ± std)")
    summary = read_result("metrics_summary")
    if summary is not None:
        show = summary[summary.model.isin(models)].copy()
        cols = ["recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc", "accuracy"]
        for m in cols:
            show[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(show[f"{m}_mean"], show[f"{m}_std"].fillna(0))]
        show["model"] = show["model"].map(_model_name)
        st.dataframe(show[["model", "arm", *cols]], use_container_width=True, hide_index=True)

    sig = read_result("significance")
    if sig is not None:
        st.subheader("Paired significance vs arm A")
        st.caption("Paired t-test over seeds (primary) and Wilcoxon signed-rank. With 5 seeds the "
                   "Wilcoxon p-value cannot fall below 0.0625. No multiple-comparison correction.")
        s = sig[(sig.metric == metric) & sig.model.isin(models)].copy()
        s["model"] = s["model"].map(_model_name)
        s["significant (p<0.05)"] = s["t_pvalue"] < 0.05
        st.dataframe(s.round(4), use_container_width=True, hide_index=True)


# =================================================================== 6. Fairness & Privacy
def _interpret_privacy(row: pd.Series) -> list[str]:
    out = []
    mia = row["mia_auc_mean"]
    if mia <= 0.55:
        out.append(f"Membership inference AUC {mia:.2f}: an attacker can barely tell training "
                   "patients from unseen ones.")
    elif mia <= 0.65:
        out.append(f"Membership inference AUC {mia:.2f}: weak but non-zero membership signal.")
    else:
        out.append(f"Membership inference AUC {mia:.2f}: notable leakage - training records "
                   "are recognisably closer to the synthetic data.")
    ratio = row["dcr_ratio_mean"]
    out.append(
        f"DCR ratio {ratio:.2f}: synthetic records are "
        + ("about as far from training records as fresh real patients are (no sign of copying)."
           if ratio >= 0.9 else "closer to training records than fresh real patients are, "
           "a possible sign of memorisation."))
    dup = row["exact_duplicates_mean"]
    out.append(f"Exact copies of training rows: {dup:.1f} on average.")
    det = row["detectability_auc_mean"]
    out.append(f"Real-vs-synthetic detectability AUC {det:.2f}: " + (
        "hard to distinguish." if det <= 0.6 else "moderately distinguishable." if det <= 0.8
        else "easily distinguishable (fidelity is limited)."))
    return out


def page_fairness_privacy() -> None:
    st.title("Fairness & Privacy")
    groups = read_result("fairness_groups")
    summary = read_result("fairness_summary")
    if groups is None or summary is None:
        missing_results_message()
        return
    tab_f, tab_p = st.tabs(["Fairness", "Privacy"])
    with tab_f:
        c1, c2 = st.columns(2)
        attr = c1.selectbox("Sensitive attribute", sorted(summary["attribute"].unique()),
                            key="fair_attr")
        model = c2.selectbox("Model", ordered(summary["model"].unique(), MODEL_ORDER),
                             format_func=_model_name, key="fair_model")
        g = groups[(groups.attribute == attr) & (groups.model == model)]
        g = g.groupby(["arm", "group"])[["recall", "accuracy", "selection_rate", "n"]].mean().reset_index()
        fig = px.bar(g, x="group", y="recall", color="arm", barmode="group",
                     color_discrete_map=ARM_COLORS, category_orders={"arm": ARM_ORDER},
                     hover_data={"accuracy": ":.3f", "selection_rate": ":.3f", "n": ":.0f"})
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(style_fig(fig, 380, f"Recall per {attr} group (mean over seeds)"),
                        use_container_width=True)
        s = summary[(summary.attribute == attr) & (summary.model == model)]
        s = s.groupby("arm")[["demographic_parity_diff", "equal_opportunity_diff"]].agg(["mean", "std"])
        s.columns = [f"{a} ({b})" for a, b in s.columns]
        s = s.reindex(ordered(s.index, ARM_ORDER))
        st.dataframe(s.round(3), use_container_width=True)
        if "A_real" in s.index:
            base = s.loc["A_real", "equal_opportunity_diff (mean)"]
            st.markdown("**Interpretation** (equal opportunity difference = gap in recall "
                        "between groups; lower is fairer):")
            for arm in s.index:
                if arm == "A_real":
                    continue
                v = s.loc[arm, "equal_opportunity_diff (mean)"]
                if pd.isna(v) or pd.isna(base):
                    continue
                verb = "narrows" if v < base - 0.02 else "widens" if v > base + 0.02 else "barely changes"
                st.markdown(f"- {arm}: {verb} the {attr} recall gap ({base:.2f} -> {v:.2f}).")
            st.caption("Test groups are small (tens of patients), so these gaps are noisy; "
                       "check the std column before drawing conclusions.")
    with tab_p:
        gq = read_result("generator_quality_agg")
        if gq is None:
            missing_results_message()
            return
        st.markdown("Each GAN is trained on the imbalanced real training set; a sample at the "
                    "training class prior is compared with (a) the training rows and (b) unseen "
                    "real test rows.")
        cols = [c for c in gq.columns if c.endswith("_mean")]
        table = gq.set_index("generator")[cols].T
        table.index = [i.replace("_mean", "") for i in table.index]
        st.dataframe(table.round(3), use_container_width=True)
        for _, row in gq.iterrows():
            st.markdown(f"**{row['generator'].upper()}**")
            for line in _interpret_privacy(row):
                st.markdown(f"- {line}")
        st.info("Synthetic data is **not automatically private**: GANs can memorise rare "
                "records, and these empirical tests are not a formal guarantee such as "
                "differential privacy.")


# =================================================================== 7. Rule Browser
def page_rules() -> None:
    st.title("Rule Browser")
    rules = get_rules_df()
    st.caption("Rules are for academic demonstration only - not medical advice.")
    c1, c2, c3 = st.columns([2, 2, 1])
    q = c1.text_input("Search", placeholder="e.g. cholesterol, R12, thal", key="rule_q")
    cats = c2.multiselect("Category", sorted(rules["category"].unique()), key="rule_cats")
    sign = c3.selectbox("Evidence", ["all", "supporting (+)", "against (-)"], key="rule_sign")
    view = rules
    if q:
        mask = view.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)
        view = view[mask]
    if cats:
        view = view[view.category.isin(cats)]
    if sign != "all":
        view = view[view.cf > 0] if sign.startswith("supporting") else view[view.cf < 0]
    st.markdown(f"{len(view)} of {len(rules)} rules")
    st.dataframe(view, use_container_width=True, hide_index=True, height=480,
                 column_config={"cf": st.column_config.NumberColumn("CF", format="%+.2f"),
                                "rationale": st.column_config.TextColumn(width="large")})

    params = read_result("expert_params")
    if params is not None:
        st.subheader("Hand-written vs data-tuned CFs (diagnosis rules)")
        st.caption("Mean tuned CF over seeds, per training arm (expert_tuned variant).")
        t = params[(params.model == "expert_tuned") & (params.param != "threshold")]
        t = t.groupby(["param", "arm"])["value"].mean().reset_index()
        hand = rules.set_index("id")["cf"]
        t = pd.concat([t, pd.DataFrame({"param": hand.index, "arm": "hand-written",
                                        "value": hand.values})[lambda d: d.param.isin(t.param)]])
        colors = {**ARM_COLORS, "hand-written": "#0b0b0b"}
        fig = px.bar(t, x="param", y="value", color="arm", barmode="group",
                     color_discrete_map=colors,
                     category_orders={"arm": ["hand-written", *ARM_ORDER]},
                     labels={"param": "rule", "value": "CF"})
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)
