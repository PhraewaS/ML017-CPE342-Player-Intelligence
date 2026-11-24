"""Task 3 feature recovery: the generator ties several columns together exactly (transactions x average value =
historical spend, discount count = rate x transactions), so missing values can be filled back in."""
import numpy as np
import pandas as pd


def restore(frame, buckets):
    out = frame.copy()
    transactions = out["total_transactions"]
    discounts = out["purchases_on_discount"]
    rate = out["discount_rate_used"]
    amount = out["avg_transaction_value"]
    history = out["historical_spending"]
    count_from_discounts = (discounts / rate.where(rate > 0)).round()
    count_from_amount = (history / amount.where(amount > 0)).round()
    # A denominator of one can mean either zero or one transaction.
    recovered_count = transactions.fillna(count_from_discounts).fillna(count_from_amount.where(count_from_amount > 1))
    denominator = recovered_count.clip(lower=1).fillna(count_from_amount)
    out["total_transactions"] = recovered_count
    out["purchases_on_discount"] = discounts.fillna((rate * denominator).round())
    out["discount_rate_used"] = rate.fillna(discounts / denominator)
    out["historical_spending"] = history.fillna(amount * denominator)
    vip = out["vip_status"]
    out["is_premium_member"] = out["is_premium_member"].fillna((vip >= 3).astype(float).where(vip.notna()))

    if buckets:
        month = out["seasonal_spending_pattern"]
        out["season_group"] = month.map({0: 2, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 2})
        out["recency_group"] = pd.cut(
            out["days_since_last_purchase"], [-np.inf, 15, 45, np.inf], labels=False, right=False
        )
        out["frequency_group"] = pd.cut(
            out["purchase_frequency"], [-np.inf, 5, 8, 10, 15, np.inf], labels=False, right=False
        )
    return out


def blend_history(frame, ratio):
    out = frame.copy()
    history = out["historical_spending"]
    previous = out["prev_month_spending"] * ratio
    out["historical_spending"] = (0.6 * history + 0.4 * previous).fillna(history)
    return out
