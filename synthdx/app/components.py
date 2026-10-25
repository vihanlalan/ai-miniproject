"""Cached data access and small UI helpers for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from synthdx.config import load_config, resolve_path
from synthdx.data import load_from_config, split_from_config
from synthdx.expert import InferenceEngine, load_knowledge_base
from synthdx.gan import SyntheticGenerator
from synthdx.palette import INK, STATUS

CSS = f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1250px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
.sdx-banner {{background: #fff6e0; border: 1px solid #f3d38a; color: #5c4400;
  padding: .55rem .9rem; border-radius: 8px; font-size: .92rem; margin-bottom: 1rem;}}
.sdx-card {{background: #ffffff; border: 1px solid rgba(11,11,11,.10); border-radius: 10px;
  padding: .9rem 1rem; height: 100%;}}
.sdx-card .label {{color: {INK['secondary']}; font-size: .8rem; text-transform: uppercase;
  letter-spacing: .04em;}}
.sdx-card .value {{font-size: 1.6rem; font-weight: 600; color: {INK['primary']};}}
.sdx-card .note {{color: {INK['muted']}; font-size: .8rem;}}
.sdx-callout {{border-left: 4px solid #2a78d6; background: #eef4fc; padding: .8rem 1rem;
  border-radius: 6px; margin: .4rem 0 1rem 0;}}
.sdx-pill {{display: inline-block; padding: .15rem .65rem; border-radius: 999px; color: #fff;
  font-weight: 600; font-size: .85rem;}}
</style>
"""

RISK_COLORS = {"low": STATUS["good"], "medium": STATUS["serious"], "high": STATUS["critical"]}
RISK_ICONS = {"low": "✔", "medium": "▲", "high": "⚠"}


def disclaimer() -> None:
    st.markdown('<div class="sdx-banner"><b>Academic demo, not for clinical use.</b> '
                "Rules, data and models are for teaching purposes only and must never be "
                "used to make medical decisions.</div>", unsafe_allow_html=True)


def stat_card(col, label: str, value: str, note: str = "") -> None:
    col.markdown(f'<div class="sdx-card"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="note">{note}</div></div>',
                 unsafe_allow_html=True)


def risk_pill(level: str) -> str:
    return (f'<span class="sdx-pill" style="background:{RISK_COLORS[level]}">'
            f'{RISK_ICONS[level]} {level.upper()} RISK</span>')


def style_fig(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor=INK["surface"], plot_bgcolor=INK["surface"],
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK["primary"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        title=dict(text=title, x=0, font=dict(size=15)) if title else None,
    )
    fig.update_xaxes(gridcolor=INK["grid"], linecolor=INK["axis"])
    fig.update_yaxes(gridcolor=INK["grid"], linecolor=INK["axis"])
    return fig


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner=False)
def get_config() -> dict:
    return load_config()


@st.cache_data(show_spinner="Loading dataset...")
def get_dataset() -> pd.DataFrame:
    return load_from_config(get_config())


@st.cache_data(show_spinner=False)
def get_split() -> dict[str, pd.DataFrame]:
    cfg = get_config()
    s = split_from_config(get_dataset(), cfg, cfg["seed"])
    return {"train_pool": s.train_pool, "train": s.train, "test": s.test}


@st.cache_resource(show_spinner=False)
def get_engine() -> InferenceEngine:
    return InferenceEngine.from_config(get_config())


@st.cache_data(show_spinner=False)
def get_rules_df() -> pd.DataFrame:
    return load_knowledge_base(get_config()["expert"]["rules_path"]).to_dataframe()


def results_dir() -> Path:
    return resolve_path(get_config()["paths"]["results_dir"])


@st.cache_data(show_spinner=False)
def read_result(name: str) -> pd.DataFrame | None:
    """Read ``results/<name>.csv`` (None when missing or empty)."""
    path = results_dir() / f"{name}.csv"
    if not path.exists() or path.stat().st_size < 2:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


@st.cache_data(show_spinner=False)
def read_metadata() -> dict | None:
    path = results_dir() / "run_metadata.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource(show_spinner="Loading cached generator...")
def get_cached_generator(name: str) -> SyntheticGenerator | None:
    path = resolve_path(get_config()["paths"]["models_dir"]) / f"{name}.pkl"
    if not path.exists():
        return None
    return SyntheticGenerator.load(path)


def missing_results_message() -> None:
    st.info("No pre-computed results found yet. Run `make experiments` "
            "(or `python -m synthdx.experiments.run`) and reload this page.")
