"""Task 3 shared features. Spending is modelled as a multiplier of each player's historical spend (the scale),
which can be recovered from other spend columns when it is missing."""
import numpy as np


TARGET = "spending_30d"
ID = "player_id"
SPENDING_COLUMNS = [
    "historical_spending",
    "prev_month_spending",
    "avg_transaction_value",
]


def features(frame, historical_to_previous):
    estimated_history = (
        frame["historical_spending"]
        .fillna(frame["total_transactions"].clip(lower=1) * frame["avg_transaction_value"])
        .fillna(frame["prev_month_spending"] * historical_to_previous)
    )
    predictors = frame.drop(columns=SPENDING_COLUMNS).assign(spend_base=estimated_history)
    scale = estimated_history.fillna(1000).clip(lower=75).to_numpy()
    return predictors, scale


def nmae(labels, predictions):
    return np.abs(labels - predictions).sum() / np.abs(labels).sum()
