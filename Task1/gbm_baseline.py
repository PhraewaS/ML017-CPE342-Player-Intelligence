"""Task 1 step 1: LightGBM (Optuna-tuned) + XGBoost, 5-fold, averaged probabilities.

Writes artifacts/task1_oof_prob.npy (labelled train rows) and artifacts/task1_test_prob.npy (test rows).
These are the baseline scores that generator_blend.py mixes with the generator model.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold

from shared.config import task_artifacts, T1

SEED, N_FOLDS = 42, 5
# Optuna result (3-fold CV on train only, see tune_lgbm_optuna.py)
LGB_TUNED = {"learning_rate": 0.023105255265581114, "num_leaves": 74, "min_child_samples": 43,
             "subsample": 0.9941869190296131, "colsample_bytree": 0.37143136752361966,
             "reg_lambda": 0.008613240311939907, "reg_alpha": 0.004418125737902547, "n_estimators": 891}


ARTIFACTS = task_artifacts(1)


def main():
    tr = pd.read_csv(T1 / 'train.csv'); te = pd.read_csv(T1 / 'test.csv')
    tr = tr[tr['is_cheater'].notna()].reset_index(drop=True)   # about 13% of train rows have no label
    feats = [c for c in te.columns if c != 'player_id']
    Xtr, Xte, y = tr[feats].astype(float), te[feats].astype(float), tr['is_cheater'].values.astype(int)

    p_lgb = dict(dict(learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.7, n_estimators=800), **LGB_TUNED, subsample_freq=1, verbose=-1, random_state=SEED)
    p_xgb = dict(n_estimators=700, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7,
                 tree_method='hist', random_state=SEED)
    oof = {m: np.zeros(len(Xtr)) for m in ('lgb', 'xgb')}
    test = {m: np.zeros(len(Xte)) for m in ('lgb', 'xgb')}
    for fold, (a, b) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xtr, y)):
        m1 = lgb.LGBMClassifier(**p_lgb).fit(Xtr.iloc[a], y[a])
        m2 = xgb.XGBClassifier(**p_xgb).fit(Xtr.iloc[a], y[a])
        for name, m in (('lgb', m1), ('xgb', m2)):
            oof[name][b] = m.predict_proba(Xtr.iloc[b])[:, 1]
            test[name] += m.predict_proba(Xte)[:, 1] / N_FOLDS
        print(f'fold {fold} done', flush=True)
    oof_p = (oof['lgb'] + oof['xgb']) / 2
    test_p = (test['lgb'] + test['xgb']) / 2
    print(f"OOF binary F2 at threshold 0.18: {fbeta_score(y, oof_p > .18, beta=2):.4f}")
    np.save(ARTIFACTS / 'task1_oof_prob.npy', oof_p)
    np.save(ARTIFACTS / 'task1_test_prob.npy', test_p)


if __name__ == '__main__':
    main()
