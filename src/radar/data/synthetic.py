from __future__ import annotations
import numpy as np
import pandas as pd


def generate_mock_data(
    start_date: str,
    end_date: str,
    tz: str = "Europe/Copenhagen",
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic day-ahead data for testing the pipeline (never for results).

    end_date is exclusive. Same seed gives the same data.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_date, end_date, freq="15min", tz=tz,
                        inclusive="left", name="timestamp")
    n = len(idx)
    hour = np.asarray(idx.hour + idx.minute / 60.0)

    # solar: daytime bell shape, exactly zero at night
    solar_shape = np.clip(np.sin((hour - 6) * np.pi / 12), 0, 1)
    solar_fc = np.clip(solar_shape * (800 + rng.normal(0, 50, n)), 0, None)

    # wind: AR(1) around 1500 MW, so neighbouring slots are correlated
    phi = 0.99
    eps = rng.normal(0, 400 * np.sqrt(1 - phi**2), n)
    wind = np.empty(n)
    wind[0] = 1500.0
    for t in range(1, n):
        wind[t] = 1500.0 + phi * (wind[t - 1] - 1500.0) + eps[t]
    wind_fc = np.clip(wind, 0, None)

    # load: smooth daily profile plus noise
    load_profile = (2500 + 500 * np.sin((hour - 5) * np.pi / 12)
                    + 300 * np.sin((hour - 15) * np.pi / 6))
    load_fc = load_profile + rng.normal(0, 100, n)

    # price: rises with residual load, heavy-tailed noise,
    # extra drop when renewables overproduce (negative-price episodes)
    residual_load = load_fc - solar_fc - wind_fc
    price = 40.0 + residual_load / 100 * 5 + rng.standard_t(3, n) * 15
    price = np.where(residual_load < -500, price - 50, price)

    return pd.DataFrame(
        {"price": price, "solar_fc": solar_fc, "wind_fc": wind_fc, "load_fc": load_fc},
        index=idx,
    )