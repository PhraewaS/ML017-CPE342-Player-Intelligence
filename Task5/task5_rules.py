"""Refine the team's anomaly predictions using each player's four time periods."""
import numpy as np


def detect_anomalies(data, original):
    def periods(feature):
        return data[[f"{feature}_{i}" for i in range(1, 5)]].to_numpy(float)

    def inside(feature, low, high):
        values = periods(feature)
        present = np.isfinite(values)
        in_band = (values >= low) & (values <= high)
        return (np.isnan(values) | in_band).all(axis=1) & present.any(axis=1)

    def incompatible(feature, noise, low, high, integer=False):
        # Bound the shared activity level implied by each observed period.
        values = periods(feature)
        noise = np.asarray(noise)
        lower = np.where(np.isnan(values), -np.inf, values / (1 + noise))
        upper = np.where(np.isnan(values), np.inf, values / (1 - noise))
        low_bound = np.maximum(low, lower.max(axis=1))
        high_bound = np.minimum(high, upper.min(axis=1))
        if integer:
            return np.ceil(low_bound - 1e-7) > np.floor(high_bound + 1e-7)
        return low_bound > high_bound + 1e-7

    high_variation = (
        inside("kda_std", 0.70, 1.00)
        & inside("accuracy_std", 9, 12)
        & inside("performance_stability", 0.85, 0.95)
    )
    inconsistent = (
        incompatible("login_count", [.20, .30, .25, .15], 20, 99, integer=True)
        | incompatible("purchase_frequency", [.50, .40, .30, .20], 0, 5)
    )
    missing_stability = data["behavior_stability"].isna().to_numpy()
    inferred_instability = missing_stability & inside("purchase_frequency", 3, 6) & inconsistent
    return (original | high_variation | inferred_instability).astype(int)
