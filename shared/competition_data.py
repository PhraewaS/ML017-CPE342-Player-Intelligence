"""Read an official competition CSV from the downloaded ZIP or extracted dataset."""
from pathlib import Path
import zipfile

import pandas as pd


def read_task(source: Path, folder: str, filename: str) -> pd.DataFrame:
    relative = Path(folder) / filename
    if source.is_dir():
        path = source / relative
        if not path.exists():
            path = source / "cpe342-kaggle-main" / relative
        return pd.read_csv(path)
    with zipfile.ZipFile(source) as archive:
        with archive.open("cpe342-kaggle-main/" + relative.as_posix()) as csv_file:
            return pd.read_csv(csv_file)
