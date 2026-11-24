"""Step 5 of the DINOv3 scene model: per-image training weights that up-weight rare visual groups.
Uses patch features cached by t4_patch_head.py --extract from the strict DINOv3 model (train images only).
Recorded from the Codex session; run from the original working folder (project root with work/review/)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, 'work/review')
from t4_patch_head import group_weights

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
work = Path('work/review')
train = pd.read_csv(root / 'data_task4/train.csv')
holdout = pd.read_csv(work / 't4_independent_split_strict.csv').set_index('id').loc[train.id, 'is_val'].to_numpy(bool)
source = np.load(work / 't4_dinov3_strict_patches.npy', allow_pickle=False, mmap_mode='r')
weights = group_weights(source, np.flatnonzero(~holdout), train.label.to_numpy()).numpy()
with (work / 't4_scene_rare_group_weights.npz').open('xb') as f:
    np.savez(f, ids=train.id.to_numpy(dtype='U'), holdout=holdout, weights=weights)
print('Wrote ordered training weights, range:', round(float(weights.min()), 3), round(float(weights.max()), 3))
