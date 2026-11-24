"""Combine the original seven unsupervised anomaly signatures."""

import numpy as np
import pandas as pd


def detect_signatures(d):
    V = lambda b, t: d[f'{b}_{t}']
    R = lambda b, t, s=1: (V(b, t) / V(b, s)).replace([np.inf, -np.inf], np.nan)
    anyp = lambda b, f: pd.concat([f(V(b, t)) for t in range(1, 5)], axis=1).any(axis=1)
    S = {
        # static / all-period types
        'bstab_lt0.7': d.behavior_stability < 0.7,
        'ip_lt0.25': anyp('ip_hash_entropy', lambda s: s < 0.25),
        'ltv_lt1': anyp('login_time_variance', lambda s: s < 1.0),
        'sdstd_lt6': anyp('session_duration_std', lambda s: s < 6),
        'kdastd_lt.15': anyp('kda_std', lambda s: s < 0.1495),
        'comm_lt.15': anyp('communication_frequency', lambda s: s < 0.15),
        'act_lt.32': anyp('activity_level', lambda s: s < 0.32),
        'guild1_eq3': V('guild_changes', 1) == 3,
        # period-4 jumps
        'dev4_+2': (V('device_count', 4) - V('device_count', 1)) >= 2,
        'guild4_+2': (V('guild_changes', 4) - V('guild_changes', 1)) >= 2,
        'frem4_gt4.8': V('friend_removes', 4) > 4.8,
        'fadd4_r>2.3': R('friend_adds', 4) > 2.3,
        'ctr4_gt12': V('currency_transfers', 4) > 12,
        'pfreq4_gt6': V('purchase_frequency', 4) > 6,
        'pamt_r43>1.6': R('purchase_amount_mean', 4, 3) > 1.6,
        'locch4_gt3.6': V('location_changes', 4) > 3.6,
        'ppc_r41<.75': R('play_pattern_consistency', 4) < 0.75,
        'rtm_r41<.81': R('reaction_time_mean', 4) < 0.81,
        'sdm_r41<.66': R('session_duration_mean', 4) < 0.66,
        'acc_r41>1.22': R('accuracy_mean', 4) > 1.22,
        'kda_r41>1.46': R('kda_mean', 4) > 1.46,
        'skill_r41>1.35': R('skill_rating', 4) > 1.35,
        'straj4_gt30': V('skill_trajectory', 4) > 30,
        'straj_r43_neg': R('skill_trajectory', 4, 3) < 0,
        'latlon_oor': (d[[f'login_lat_{t}' for t in range(1, 5)]].abs().max(axis=1) > 61.2) | (d[[f'login_lon_{t}' for t in range(1, 5)]].abs().max(axis=1) > 182),
        'loc_move3': np.hypot(V('login_lat', 3) - V('login_lat', 1), V('login_lon', 3) - V('login_lon', 1)) > 3.5,
        'loc_move4': np.hypot(V('login_lat', 4) - V('login_lat', 1), V('login_lon', 4) - V('login_lon', 1)) > 3.5,
    }
    S = pd.DataFrame({k: v.fillna(False).values for k, v in S.items()})

    # Group signatures into the seven anomaly types (dynamic types need two matching period-4 signals).
    T = pd.DataFrame({
        'bot': S[['ip_lt0.25', 'ltv_lt1', 'sdstd_lt6']].any(axis=1),
        'bstab': S['bstab_lt0.7'],
        'comm': S['comm_lt.15'] | S['guild1_eq3'],
        'act': S['act_lt.32'],
        'boost': S[['frem4_gt4.8', 'rtm_r41<.81', 'acc_r41>1.22', 'kda_r41>1.46', 'skill_r41>1.35', 'straj4_gt30', 'guild4_+2']].sum(axis=1) >= 2,
        'theft': S[['ctr4_gt12', 'ppc_r41<.75', 'sdm_r41<.66', 'latlon_oor', 'loc_move4']].sum(axis=1) >= 2,
        'vpnbuy': S[['pfreq4_gt6', 'pamt_r43>1.6', 'loc_move3']].sum(axis=1) >= 2,
    })
    # Two signatures are threshold artifacts inside the normal range, so they are excluded.
    WEAK = ['kdastd_lt.15', 'fadd4_r>2.3']
    return (T.any(axis=1).to_numpy() | S.drop(columns=WEAK).any(axis=1).to_numpy()).astype(np.int64)
