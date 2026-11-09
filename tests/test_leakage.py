import pandas as pd
import pytest

from radar.data.splits import split_for_day
from radar.data.synthetic import generate_mock_data

TZ = "Europe/Copenhagen"
PAST_ONLY = ["price"]
KNOWN_FUTURE = ["solar_fc", "wind_fc", "load_fc"]


@pytest.fixture(scope="module")
def table():
    return generate_mock_data("2026-03-20", "2026-11-05", tz=TZ, seed=1)


@pytest.mark.parametrize(
    "day, delivery_day, expected_slots",
    [
        ("2026-10-12", "2026-10-13", 96),   # normal day
        ("2026-10-24", "2026-10-25", 100),  # delivery day has the clock going back
        ("2026-03-28", "2026-03-29", 92),   # delivery day has the clock going forward
    ],
)
def test_split_boundaries(table, day, delivery_day, expected_slots):
    history, horizon = split_for_day(table, pd.Timestamp(day), PAST_ONLY, KNOWN_FUTURE)
    delivery_start = pd.Timestamp(f"{delivery_day} 00:00", tz=TZ)

    # history stops one slot before the delivery day and is untouched
    assert history.index.max() == delivery_start - pd.Timedelta(minutes=15)
    pd.testing.assert_frame_equal(history, table.loc[history.index])

    # horizon is exactly the delivery day, known-future columns only
    assert horizon.index.min() == delivery_start
    assert len(horizon) == expected_slots
    assert list(horizon.columns) == KNOWN_FUTURE
    pd.testing.assert_frame_equal(horizon, table.loc[horizon.index, KNOWN_FUTURE])

    # tomorrow's price never reaches the model
    assert not set(PAST_ONLY) & set(horizon.columns)
    assert history.index.max() < horizon.index.min()


def test_every_day_is_a_clean_split(table):
    for day in pd.date_range("2026-03-22", "2026-11-02", freq="D"):
        history, horizon = split_for_day(table, day, PAST_ONLY, KNOWN_FUTURE)
        assert len(horizon) in (92, 96, 100), day
        assert horizon.index.is_unique, day
        assert history.index.max() < horizon.index.min(), day


def test_overlapping_column_types_are_rejected(table):
    with pytest.raises(ValueError):
        split_for_day(table, pd.Timestamp("2026-10-12"), ["price"], ["price", "wind_fc"])


def test_naive_index_is_rejected(table):
    naive = table.tz_localize(None)
    with pytest.raises(ValueError):
        split_for_day(naive, pd.Timestamp("2026-10-12"), PAST_ONLY, KNOWN_FUTURE)