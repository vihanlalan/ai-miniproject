# Architecture

```mermaid
flowchart LR
    subgraph data["synthdx/data"]
        L["loader.py<br/>download + cache + offline fallback"] --> C["clean: impute, binarise target"]
        C --> S["splits.py<br/>stratified test hold-out<br/>+ imbalance simulation"]
    end

    S -->|imbalanced real train| G
    S -->|imbalanced real train| ARMS
    S -->|real test, untouched| EV

    subgraph gan["synthdx/gan"]
        G["CTGANGenerator (ctgan lib)<br/>ConditionalGAN (PyTorch)<br/>fit / generate(n, condition)"]
    end

    G -->|synthetic minority / full synthetic| ARMS

    subgraph exp["synthdx/experiments"]
        ARMS["arms.py<br/>A real · B +GAN · C +SMOTE-NC · D TSTR"]
        R["runner.py<br/>seeds × arms × models"]
        REP["report.py<br/>figures + report.md"]
    end

    ARMS --> ML
    ARMS -->|calibrate / tune| ES

    subgraph ml["synthdx/ml"]
        ML["LogReg · RandomForest · GradBoost"]
    end

    subgraph expert["synthdx/expert"]
        KB["rules.yaml (41 rules)"] --> ES
        F["facts.py<br/>working memory + fuzzy facts"] --> ES
        ES["engine.py<br/>forward chaining + MYCIN CFs"] --> X["explain.py<br/>how / why-not"]
        T["tuning.py<br/>threshold + rule-CF calibration"] --> ES
    end

    ML --> EV
    ES --> EV
    G -->|quality sample| EV

    subgraph eval["synthdx/evaluation"]
        EV["metrics · fidelity · privacy · fairness · stats"]
    end

    EV --> R --> REP
    R -->|results/*.csv| UI
    G -->|models/*.pkl| UI
    ES --> UI

    subgraph app["synthdx/app"]
        UI["Streamlit: 7 pages<br/>works offline from cached results"]
    end
```

## Module responsibilities

| Package | Responsibility | Key entry points |
|---|---|---|
| `synthdx.data` | load, clean, split, imbalance | `load_heart_disease`, `prepare_splits` |
| `synthdx.gan` | synthetic data generators | `CTGANGenerator`, `ConditionalGAN`, `build_generator` |
| `synthdx.expert` | rule-based diagnosis | `InferenceEngine.diagnose`, `format_report`, `fit_expert` |
| `synthdx.ml` | baseline classifiers | `train_classifier`, `predict_scores` |
| `synthdx.evaluation` | all metrics | `classification_metrics`, `fidelity_summary`, `privacy_summary`, `group_metrics` |
| `synthdx.experiments` | orchestration and reporting | `run_experiments`, `generate_report`, `train_demo_models` |
| `synthdx.app` | user interface | `streamlit run synthdx/app/main.py` |

All randomness flows from explicit seeds (`synthdx.utils.set_seed`, per-model
`random_state`, a private RNG inside each generator). All settings live in
`config.yaml`.
