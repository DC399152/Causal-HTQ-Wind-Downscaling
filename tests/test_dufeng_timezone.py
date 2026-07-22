import pandas as pd

from scripts.standardize_dufeng_excel import _source_times_to_utc_naive


def test_dufeng_beijing_time_is_converted_to_utc_naive():
    converted = _source_times_to_utc_naive(
        pd.Series(["2026-04-19 00:00:00", "2026-04-19 08:10:00"]),
        "Asia/Shanghai",
    )

    assert converted.dt.tz is None
    assert converted.tolist() == [
        pd.Timestamp("2026-04-18 16:00:00"),
        pd.Timestamp("2026-04-19 00:10:00"),
    ]
