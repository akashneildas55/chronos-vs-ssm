from __future__ import annotations
import pandas as pd


def split_for_day(
    df: pd.DataFrame,
    day: pd.Timestamp,
    past_only: list[str],
    known_future: list[str],
    tz: str = "Europe/Copenhagen",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut the table for the auction held on `day` (delivery day is day + 1).

    history: every row before the delivery day starts, all columns.
    horizon: every row of the delivery day, known_future columns only.
    """
    if df.index.tz is None:
        raise ValueError("df index must be timezone-aware")
    if set(past_only) & set(known_future):
        raise ValueError("a column cannot be both past_only and known_future")

    day = pd.Timestamp(day)
    day = day.tz_localize(tz) if day.tzinfo is None else day.tz_convert(tz)

    delivery_start = day.normalize() + pd.DateOffset(days=1)
    delivery_end = delivery_start + pd.DateOffset(days=1)

    history = df[df.index < delivery_start]
    in_delivery_day = (df.index >= delivery_start) & (df.index < delivery_end)
    horizon = df.loc[in_delivery_day, known_future]
    return history, horizon