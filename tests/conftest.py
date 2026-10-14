import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_shipments(rows):
    """rows: iterable of (route, route_type, date, tonnes, km, cost). Returns a cleaned frame."""
    from src.ingestion import clean_shipments

    recs = []
    for i, (route, rtype, date, q, d, c) in enumerate(rows):
        o, dest = route.split("-")
        recs.append({
            "shipment_id": f"S{i:05d}", "origin": o, "destination": dest, "route_type": rtype,
            "material": "Steel", "quantity_tonnes": str(q), "distance_km": str(d),
            "freight_cost_inr": str(c), "shipment_date": date, "transporter": "T",
        })
    df, report = clean_shipments(pd.DataFrame(recs))
    assert report.rows_kept == len(recs), report.to_dict()
    return df


@pytest.fixture(scope="session")
def tfidf_run():
    """One full pipeline run on the supplied data with the lexical backend (no model download)."""
    from src.pipeline import run_pipeline

    return run_pipeline(retriever="tfidf")
