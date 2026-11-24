"""Dataset, cache, and submission layout shared by the task scripts."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CPE342_DATA", ROOT / "data/cpe342-kaggle-main"))
T4_DATA = Path(os.environ.get("CPE342_T4_DATA", ROOT / "data/data_task4"))
T1 = DATA / "01_binary_classification"
T2 = DATA / "02_multiclass_classification"
T3 = DATA / "03_regression"
ROWS = 25_889
IMAGE_CLASSES = 5


def task_artifacts(task):
    folder = Path(os.environ.get("CPE342_ARTIFACTS", ROOT / f"Task{task}/cache"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def sample_ids():
    return [f"ANS{row:05d}" for row in range(1, ROWS + 1)]
