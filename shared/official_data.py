"""Check row alignment between the official files and the cached v3 predictions."""
import hashlib
from pathlib import Path

from .competition_data import read_task

ID_HASHES = {
    ("01_binary_classification", "test.csv"): "757cc75b90d1fe9e3e41674df7bf2dbb5db97fbf25430593424d79cd4a4ec99b",
    ("01_binary_classification", "train.csv"): "6f31a3e9f4e5558a22678d7a993b551b0de75f08e059221d6adb04fc83e012cb",
    ("03_regression", "test.csv"): "d17c40ff27478d311ea47c86deaa529001a748fce96ca258d785452372639008",
    ("03_regression", "train.csv"): "107d6b9c03613119c49b8245a276ad5af26a41c3cea38ab0aa948fdc4c87b4ed",
}


def read_aligned(source, folder, name):
    frame = read_task(source, folder, name)
    digest = hashlib.sha256("\n".join(frame.player_id).encode()).hexdigest()
    if digest != ID_HASHES[folder, name]:
        raise ValueError(f"The player order in {folder}/{name} differs from the v3 cache.")
    return frame
