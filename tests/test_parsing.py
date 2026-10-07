import pandas as pd

from fruitfresh.data_index import parse_filename


def test_original_screenshot():
    r = parse_filename("Screen Shot 2018-06-08 at 5.18.26 PM.png")
    assert r["is_original"] and r["aug_type"] == ""
    assert r["capture_time"] == pd.Timestamp("2018-06-08 17:18:26")
    assert r["base_name"] == "Screen Shot 2018-06-08 at 5.18.26 PM.png"


def test_augmented_prefixes():
    r = parse_filename("rotated_by_15_Screen Shot 2018-06-07 at 12.05.01 AM.png")
    assert not r["is_original"] and r["aug_type"] == "rotated_by_15"
    assert r["base_name"] == "Screen Shot 2018-06-07 at 12.05.01 AM.png"
    assert r["capture_time"] == pd.Timestamp("2018-06-07 00:05:01")
    for prefix in ("vertical_flip", "translation", "saltandpepper"):
        assert parse_filename(f"{prefix}_Screen Shot 2018-06-12 at 1.02.03 PM.png")["aug_type"] == prefix


def test_chained_prefixes_and_noon():
    r = parse_filename("vertical_flip_rotated_by_30_Screen Shot 2018-06-12 at 12.30.00 PM.png")
    assert r["aug_type"] == "vertical_flip+rotated_by_30"
    assert r["capture_time"] == pd.Timestamp("2018-06-12 12:30:00")


def test_non_screenshot_has_no_time():
    r = parse_filename("IMG_0001.jpg")
    assert r["is_original"] and pd.isna(r["capture_time"])


def test_narrow_nbsp_before_pm():
    r = parse_filename("Screen Shot 2018-06-08 at 5.18.26\u202fPM.png")
    assert r["capture_time"] == pd.Timestamp("2018-06-08 17:18:26")
