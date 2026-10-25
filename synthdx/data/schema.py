"""Schema of the UCI Heart Disease (Cleveland) dataset.

Feature meanings (UCI documentation):
    age       age in years
    sex       1 = male, 0 = female
    cp        chest pain type: 1 typical angina, 2 atypical angina,
              3 non-anginal pain, 4 asymptomatic
    trestbps  resting blood pressure (mm Hg)
    chol      serum cholesterol (mg/dl)
    fbs       fasting blood sugar > 120 mg/dl (1 = true)
    restecg   resting ECG: 0 normal, 1 ST-T wave abnormality, 2 LV hypertrophy
    thalach   maximum heart rate achieved
    exang     exercise-induced angina (1 = yes)
    oldpeak   ST depression induced by exercise relative to rest
    slope     slope of peak exercise ST segment: 1 up, 2 flat, 3 down
    ca        number of major vessels (0-3) coloured by fluoroscopy
    thal      3 normal, 6 fixed defect, 7 reversible defect
    target    1 = angiographic disease present (num > 0), 0 = absent
"""

from __future__ import annotations

RAW_COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

TARGET: str = "target"
NUMERIC_COLUMNS: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS: list[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS: list[str] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
ALL_COLUMNS: list[str] = FEATURE_COLUMNS + [TARGET]

#: Allowed levels for every categorical column.
CATEGORY_LEVELS: dict[str, list[int]] = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
}

#: Physiologically plausible bounds used to clip generated values.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "trestbps": (80, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0.0, 7.0),
}

#: Numeric columns stored as integers (oldpeak keeps one decimal).
INTEGER_NUMERIC_COLUMNS: list[str] = ["age", "trestbps", "chol", "thalach"]

FEATURE_LABELS: dict[str, str] = {
    "age": "Age (years)",
    "sex": "Sex (1 = male)",
    "cp": "Chest pain type",
    "trestbps": "Resting BP (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate",
    "exang": "Exercise angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST slope",
    "ca": "Major vessels (fluoroscopy)",
    "thal": "Thallium scan",
    TARGET: "Heart disease",
}
