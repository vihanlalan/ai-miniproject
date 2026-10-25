"""Cross-platform equivalent of the Makefile (for machines without `make`).

    python tasks.py test | test-fast | train | experiments | report | demo | all
"""

from __future__ import annotations

import subprocess
import sys

PY = sys.executable
TARGETS: dict[str, list[list[str]]] = {
    "install": [[PY, "-m", "pip", "install", "-r", "requirements.txt"]],
    "test": [[PY, "-m", "pytest", "--cov=synthdx", "--cov-report=term-missing"]],
    "test-fast": [[PY, "-m", "pytest", "-m", "not slow"]],
    "train": [[PY, "-m", "synthdx.experiments.train"]],
    "experiments": [[PY, "-m", "synthdx.experiments.run"]],
    "report": [[PY, "-m", "synthdx.experiments.report"]],
    "demo": [[PY, "-m", "streamlit", "run", "synthdx/app/main.py"]],
}
TARGETS["all"] = TARGETS["train"] + TARGETS["experiments"] + TARGETS["report"] + TARGETS["test"]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in TARGETS:
        sys.stderr.write(f"usage: python tasks.py [{' | '.join(TARGETS)}]\n")
        return 2
    for cmd in TARGETS[sys.argv[1]]:
        code = subprocess.call(cmd)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
