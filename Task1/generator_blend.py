"""Task 1 step 2: generative ("account generator") model blended with the GBM baseline.

EDA showed the cheater/legit accounts come from a few components with uniform supports on account age, friend
network and reports received, a deterministic level / age / progression-speed relation, and Gaussian-dependent
combat statistics. This script fits that class-conditional mixture on train only (missing values integrated out),
checks it out-of-fold, then predicts test as 0.25 * GBM + 0.75 * generator and thresholds at 0.18 (the binary-F2
optimum on OOF; the leaderboard scores binary F2). Writes artifacts/task1.npy.
Run Task1/gbm_baseline.py first.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit, logsumexp, ndtri, stdtrit
from scipy.stats import multivariate_t, t as student_t
from sklearn.metrics import fbeta_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from shared.config import task_artifacts, ROWS, T1

THRESHOLD = 0.18
GENERATOR_WEIGHT = 0.75
RULE = 'dependent_combat'   # Gaussian copula between combat features


AGE, LEVEL, SPEED, FRIENDS, REPORTS, AIM = (
    'account_age_days', 'level', 'level_progression_speed', 'friend_network_size',
    'reports_received', 'crosshair_placement',
)
# Each component is (target, age support, friend support, report support).
COMPONENTS = [
    (0, (1, 120), (0, 49), (10, 29)),
    (0, (120, 900), (45, 249), (5, 19)),
    (0, (350, 1300), (120, 349), (8, 24)),
    (0, (600, 1500), (100, 319), (5, 17)),
    (1, (15, 180), (10, 89), (4, 19)),
    (1, (200, 1000), (60, 269), (4, 19)),
]
CHEATER_REPORT_COMPONENTS = [
    (1, ages, friends, report)
    for _, ages, friends, _ in COMPONENTS[4:]
    for report in ((4, 15), (6, 19))
]
LEVELS = [(28, 100), (25, 96)]
AIMS = [(0.82, 0.96), (0.80, 0.94)]
COMBAT = [
    'kill_death_ratio', 'headshot_percentage', 'game_sense_score', 'accuracy_score',
    'spray_control_score', 'aiming_smoothness', 'win_rate', 'kill_consistency',
    'reaction_time_ms',
]
SPEED_MIN, SPEED_MAX = 0.15, 1.08
NODES, WEIGHTS = leggauss(24)


def uniform(x, limits, discrete=False):
    lo, hi = limits
    result = ((x >= lo) & (x <= hi)) / (hi - lo + int(discrete))
    return np.where(np.isnan(x), 1.0, result)


def speed_density(age, level, speed, age_limits, level_limits):
    """Integrate out missing age and level from p(age, level, speed)."""
    a0, a1 = age_limits
    l0, l1 = level_limits
    width = SPEED_MAX - SPEED_MIN
    n_age, n_level, n_speed = np.isnan(age), np.isnan(level), np.isnan(speed)
    density = uniform(age, age_limits) * uniform(level, level_limits)

    k = ~n_speed & ~n_age & ~n_level
    ratio = speed[k] * age[k] / level[k]
    within = (ratio >= SPEED_MIN - 1e-9) & (ratio <= SPEED_MAX + 1e-9)
    density[k] *= within * age[k] / (width * level[k])

    k = ~n_speed & n_age & ~n_level
    lo = np.maximum(a0, SPEED_MIN * level[k] / speed[k])
    hi = np.minimum(a1, SPEED_MAX * level[k] / speed[k])
    density[k] *= np.maximum(0, hi**2 - lo**2) / (2 * level[k] * width * (a1 - a0))

    k = ~n_speed & ~n_age & n_level
    lo = np.maximum(l0, age[k] * speed[k] / SPEED_MAX)
    hi = np.minimum(l1, age[k] * speed[k] / SPEED_MIN)
    density[k] *= age[k] * np.maximum(0, np.log(hi / lo)) / (width * (l1 - l0))

    k = ~n_speed & n_age & n_level
    quadrature_level = (l1 + l0 + (l1 - l0) * NODES) / 2
    lo = np.maximum(a0, SPEED_MIN * quadrature_level / speed[k, None])
    hi = np.minimum(a1, SPEED_MAX * quadrature_level / speed[k, None])
    partial = np.maximum(0, hi**2 - lo**2) / (2 * quadrature_level * width * (a1 - a0))
    density[k] = partial @ (WEIGHTS / 2)
    return density


def population_features(frame):
    age, level, speed, friends, reports, aim = [frame[c].to_numpy() for c in [AGE, LEVEL, SPEED, FRIENDS, REPORTS, AIM]]
    result = []
    for y, ages, fr, rep in COMPONENTS:
        joint = speed_density(age, level, speed, ages, LEVELS[y])
        joint *= uniform(friends, fr, discrete=True) * uniform(aim, AIMS[y])
        if y == 1 and rep == (4, 19):
            # The breakpoints in cheater report counts are shared across the two age groups.
            joint *= (0.69 * uniform(reports, (4, 15), discrete=True)
                      + 0.31 * uniform(reports, (6, 19), discrete=True))
        else:
            joint *= uniform(reports, rep, discrete=True)
        result.append(joint)
    return np.column_stack(result)


def combat_codes(train, evaluate, point_masses=False):
    tr_codes, ev_codes, sizes, continuous_slices = [], [], [], []
    for column in COMBAT:
        fit = train[column].to_numpy()
        lower, upper = np.nanmin(fit), np.nanmax(fit)
        low_atom = int(point_masses and lower < upper and np.count_nonzero(fit == lower) >= 30)
        high_atom = int(point_masses and lower < upper and np.count_nonzero(fit == upper) >= 30)
        continuous = fit
        if low_atom:
            continuous = continuous[continuous != lower]
        if high_atom:
            continuous = continuous[continuous != upper]
        if not np.isfinite(continuous).any():
            continuous = fit
            low_atom = high_atom = 0
        edges = np.unique(np.nanquantile(continuous, np.linspace(0, 1, 41)))[1:-1]
        size = len(edges) + 1 + low_atom + high_atom
        codes = []
        for frame in [train, evaluate]:
            values = frame[column].to_numpy()
            code = np.searchsorted(edges, values, side='right') + low_atom
            if low_atom:
                code[values == lower] = 0
            if high_atom:
                code[values == upper] = size - 1
            codes.append(np.where(np.isnan(values), size, code))
        tr_codes.append(codes[0])
        ev_codes.append(codes[1])
        sizes.append(size)
        continuous_slices.append(slice(low_atom, size - high_atom))
    return tr_codes, ev_codes, sizes, continuous_slices


def smooth_counts(counts, continuous):
    probability = counts + 1
    probability[continuous] = gaussian_filter1d(probability[continuous], 1.25)
    return probability / probability.sum()


def combat_log_density(train_codes, target_codes, sizes, continuous_slices, membership, labels, use_components):
    out = np.zeros((len(target_codes[0]), membership.shape[1]))
    for fit, ev, size, continuous in zip(train_codes, target_codes, sizes, continuous_slices, strict=True):
        class_counts = [np.bincount(fit, weights=(labels == y), minlength=size + 1)[:size] for y in (0, 1)]
        for i, (y, *_) in enumerate(COMPONENTS):
            pooled = class_counts[y] / class_counts[y].sum()
            if use_components:
                counts = np.bincount(fit, weights=membership[:, i], minlength=size + 1)[:size]
                counts += 1000 * pooled
            else:
                counts = class_counts[y]
            logs = np.r_[np.log(smooth_counts(counts, continuous)), 0.0]
            out[:, i] += logs[ev]
    return out


def correction(train_codes, target_codes, sizes, continuous_slices, membership, labels, tail_df=None):
    """Estimate the dependence between within-component percentile ranks."""
    out = np.zeros((len(target_codes[0]), len(COMPONENTS)))
    available = np.column_stack([code != size for code, size in zip(target_codes, sizes, strict=True)])
    # Rows sharing a missingness pattern use the same marginal covariance.
    status = available @ (1 << np.arange(len(sizes)))
    patterns = [(status == k, np.flatnonzero(k & (1 << np.arange(len(sizes))))) for k in np.unique(status)]
    for i, (y, *_) in enumerate(COMPONENTS):
        fit_z, score_z = [], []
        weights = membership[:, i]
        for fit, ev, size, continuous in zip(train_codes, target_codes, sizes, continuous_slices, strict=True):
            counts = np.bincount(fit, weights=weights, minlength=size + 1)[:size]
            pooled = np.bincount(fit, weights=(labels == y), minlength=size + 1)[:size]
            p = smooth_counts(counts + 1000 * pooled / pooled.sum(), continuous)
            percentile = np.cumsum(p) - p / 2
            z = np.r_[ndtri(percentile) if tail_df is None else stdtrit(tail_df, percentile), np.nan]
            fit_z.append(z[fit])
            score_z.append(z[ev])
        fit_z = np.column_stack(fit_z)
        present = np.isfinite(fit_z).astype(float)
        means = (np.nan_to_num(fit_z) * weights[:, None]).sum(axis=0) / (present * weights[:, None]).sum(axis=0)
        centered = np.nan_to_num(fit_z - means)
        pair_weights = present.T @ (present * weights[:, None])
        cov = (centered.T @ (centered * weights[:, None])) / pair_weights
        inv_sd = 1 / np.sqrt(np.diag(cov))
        rho = 0.9 * cov * inv_sd[:, None] * inv_sd[None, :] + 0.1 * np.eye(len(sizes))
        score_z = np.column_stack(score_z)
        for rows, columns in patterns:
            if len(columns) < 2:
                continue
            r = rho[np.ix_(columns, columns)]
            z = score_z[np.ix_(rows, columns)]
            if tail_df is None:
                adjustment = np.einsum('ij,ij->i', z @ (np.linalg.inv(r) - np.eye(len(columns))), z)
                out[rows, i] = -0.5 * (adjustment + np.linalg.slogdet(r)[1])
            else:
                out[rows, i] = multivariate_t.logpdf(z, shape=r, df=tail_df) - student_t.logpdf(z, tail_df).sum(axis=1)
    return np.clip(out, -5, 5)


def mixture(train, evaluate, ya, pop_train, pop_evaluate, variant, point_masses=False, tail_df=None):
    class_index = np.array([c[0] for c in COMPONENTS])
    ok = ya[:, None] == class_index
    valid = (pop_train * ok).sum(axis=1) > 0
    # Protect evaluation against any rows outside the inferred generator.
    pop_train = pop_train[valid]
    ok = ok[valid]
    ya = ya[valid]
    train = train.iloc[np.flatnonzero(valid)]
    with np.errstate(divide='ignore'):
        log_pop_train = np.where(ok, np.log(pop_train), -np.inf)
        log_pop_evaluate = np.log(pop_evaluate)
    p = np.bincount(class_index)[class_index].astype(float)
    prior = np.bincount(ya)[class_index] / len(ya)
    weights = prior / p
    log_combat = np.zeros_like(log_pop_train)
    if variant != 'population':
        t, e, sizes, continuous = combat_codes(train, evaluate, point_masses)

    for i in range(8):
        log_joint = log_pop_train + log_combat + np.log(weights)
        responsibility = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
        weights = responsibility.mean(axis=0)
        if variant in ('component_combat', 'dependent_combat'):
            log_combat = combat_log_density(t, t, sizes, continuous, responsibility, ya, use_components=True)

    test_joint = log_pop_evaluate + np.log(weights)
    if variant != 'population':
        test_joint += combat_log_density(t, e, sizes, continuous, responsibility, ya, use_components=variant != 'class_combat')
    if variant == 'dependent_combat':
        test_joint += correction(t, e, sizes, continuous, responsibility, ya, tail_df)
    can_score = np.isfinite(test_joint).any(axis=1)
    logits = np.full(len(evaluate), np.nan)
    logits[can_score] = logsumexp(test_joint[can_score][:, class_index == 1], axis=1) - logsumexp(
        test_joint[can_score][:, class_index == 0], axis=1,
    )
    return expit(logits), weights


def f2_grid(y, p):
    ts = np.arange(0.14, 0.281, 0.005)
    counts = (p[:, None] >= ts)
    true_pos = (counts & (y[:, None] == 1)).sum(axis=0)
    scores = 5 * true_pos / (4 * y.sum() + counts.sum(axis=0))
    return ts, scores


ARTIFACTS = task_artifacts(1)


def main():
    source = pd.read_csv(T1 / 'train.csv').dropna(subset='is_cheater')
    y = source['is_cheater'].to_numpy(dtype=int)
    features = source.drop(columns=['is_cheater', 'player_id'])
    baseline = np.load(ARTIFACTS / 'task1_oof_prob.npy', allow_pickle=False)
    population = population_features(features)

    # Out-of-fold check of the same recipe (train only).
    generator_oof = np.zeros(len(source))
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(features, y)):
        pred, weight = mixture(features.iloc[a], features.iloc[b], y[a], population[a], population[b], RULE)
        generator_oof[b] = np.where(np.isnan(pred), baseline[b], pred)
        print(f'fold {fold}: generator F2@{THRESHOLD} = {fbeta_score(y[b], generator_oof[b] > THRESHOLD, beta=2):.5f}', flush=True)
    blend_oof = (1 - GENERATOR_WEIGHT) * baseline + GENERATOR_WEIGHT * generator_oof
    for name, pred in (('GBM baseline', baseline), ('generator', generator_oof), ('blend', blend_oof)):
        print(f'{name:13s} OOF binary F2@{THRESHOLD}: {fbeta_score(y, pred > THRESHOLD, beta=2):.5f}  AUC {roc_auc_score(y, pred):.5f}', flush=True)

    test = pd.read_csv(T1 / 'test.csv')
    if len(test) != ROWS or not test['player_id'].is_unique or test['player_id'].isna().any():
        raise ValueError('Unexpected Task 1 test file.')
    test_features = test.drop(columns='player_id')
    if not test_features.columns.equals(features.columns):
        raise ValueError('Task 1 test feature columns differ from training.')
    test_baseline = np.load(ARTIFACTS / 'task1_test_prob.npy', allow_pickle=False)
    trained, _ = mixture(features, test_features, y, population, population_features(test_features), RULE)
    generator_scores = np.where(np.isnan(trained), test_baseline, trained)
    blend = (1 - GENERATOR_WEIGHT) * test_baseline + GENERATOR_WEIGHT * generator_scores
    if not np.isfinite(blend).all():
        raise ValueError('Unexpected nonfinite scores.')
    result = (blend > THRESHOLD).astype(np.int64)
    np.save(ARTIFACTS / 'task1.npy', result)
    print(f'Wrote {len(result):,} Task 1 labels ({result.mean():.1%} flagged).', flush=True)


if __name__ == '__main__':
    main()
