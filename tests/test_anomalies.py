import math

import pandas as pd
import pytest

from src.anomaly_detection import classify_numeric, detect_anomalies
from src.config import Config

CFG = Config()  # 15% own history, 20% peers, must be rising


@pytest.mark.parametrize("own, peer, anomaly, triggers, status", [
    (35.5, 21.0, True, ["own_history", "similar_routes"], "anomaly"),  # sample row 1
    (9.2, 23.6, True, ["similar_routes"], "anomaly"),                   # sample row 3: peers only
    (16.0, 0.0, True, ["own_history"], "anomaly"),
    (15.0, 0.0, True, ["own_history"], "anomaly"),                      # threshold is inclusive
    (14.99, 19.99, False, [], "within_thresholds"),
    (-0.09, 21.4, False, [], "not_rising"),                             # high vs peers but not rising
    (0.0, 50.0, False, [], "not_rising"),
    (math.nan, 50.0, False, [], "no_history"),                          # first week of a route
    (16.0, math.nan, True, ["own_history"], "anomaly"),                 # no peers that week
    (5.0, math.nan, False, [], "within_thresholds"),
])
def test_classify_numeric(own, peer, anomaly, triggers, status):
    assert classify_numeric(own, peer, CFG) == (anomaly, triggers, status)


def test_thresholds_are_configurable():
    strict = Config(own_history_threshold_pct=30, peer_threshold_pct=40)
    assert classify_numeric(29.0, 39.0, strict)[0] is False
    loose = Config(require_rising=False)
    assert classify_numeric(-1.0, 25.0, loose)[0] is True


def test_detect_anomalies_is_deterministic():
    df = pd.DataFrame({"own_history_pct": [20.0, 1.0, math.nan, 10.0],
                       "peer_pct": [0.0, 30.0, 40.0, 5.0]})
    a, b = detect_anomalies(df, CFG), detect_anomalies(df, CFG)
    pd.testing.assert_frame_equal(a, b)
    assert a["numeric_anomaly"].tolist() == [True, True, False, False]
    assert a["anomaly_triggers"].tolist() == ["own_history", "similar_routes", "", ""]
