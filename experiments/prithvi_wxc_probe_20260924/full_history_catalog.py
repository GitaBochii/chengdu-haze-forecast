"""Public CMR metadata inventory; never authenticates or downloads science data."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent / "full_history"
PRODUCTS = ("M2I1NXASM", "M2T1NXLND", "M2T1NXFLX", "M2T1NXRAD", "M2I3NVASM")
CMR = "https://cmr.earthdata.nasa.gov/search/"


def select_required_granules(granules, required_dates):
    return [g for g in granules if pd.Timestamp(g["time_start"]).tz_convert("UTC").date().isoformat() in required_dates]


def build_coverage(required_dates, products):
    required = set(required_dates)
    covered_all = set(required)
    summaries = {}
    for name, product in products.items():
        dates = {pd.Timestamp(g["time_start"]).tz_convert("UTC").date().isoformat() for g in product["granules"]} & required
        absent = sorted(required - dates)
        complete = product["complete"]
        summaries[name] = {
            "catalog_query_complete": complete,
            "nominal_dates_with_metadata": len(dates),
            "granule_count": len(product["granules"]),
            "missing_nominal_dates": absent if complete else [],
            "unresolved_nominal_dates": absent if not complete else [],
            "years": {
                year: {
                    "required_nominal_dates": sum(d.startswith(year) for d in required),
                    "nominal_dates_with_metadata": sum(d.startswith(year) for d in dates),
                }
                for year in sorted({d[:4] for d in required})
            },
        }
        covered_all &= dates
    return {
        "dependency_scope": "nominal-required UTC dates only",
        "data_downloaded": False,
        "preprocessing_complete": False,
        "required_nominal_dates": len(required),
        "products": summaries,
        "all_products_metadata_dates": sorted(covered_all),
        "limitations": [
            "Metadata existence does not prove authorized access, usable fields or complete intra-day coverage.",
            "Adjacent-day averaged-field support, static fields and climatology are not inventoried here.",
            "This is retrospective reanalysis metadata, not operational availability evidence.",
        ],
    }


def public_json(session, endpoint, params):
    for attempt in range(4):
        try:
            response = session.get(CMR + endpoint, params=params, timeout=(20, 120))
            response.raise_for_status()
            return response.json(), response.headers
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main():
    source = BASE / "required_weather_times.parquet"
    times = pd.to_datetime(pd.read_parquet(source)["weather_time_utc"], utc=True)
    dates = sorted(set(times.dt.date.astype(str)))
    out = BASE / "catalog"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    # Explicitly disable netrc/environment credential discovery for this public inventory.
    session.trust_env = False
    session.headers["User-Agent"] = "chengdu-haze-public-catalog/1.0"
    products = {}
    for name in PRODUCTS:
        product = {"short_name": name, "version": "5.12.4", "complete": False, "granules": []}
        products[name] = product
        try:
            data, _ = public_json(session, "collections.json", {"short_name": name, "version": "5.12.4", "page_size": 100})
            collections = [e for e in data["feed"]["entry"] if e.get("data_center") == "GES_DISC" and e.get("short_name") == name]
            if len(collections) != 1:
                raise ValueError("Expected one exact GES DISC collection")
            collection = collections[0]
            product["collection"] = collection
            seen = set()
            count = 0
            page = 1
            while True:
                data, headers = public_json(session, "granules.json", {
                    "collection_concept_id": collection["id"],
                    "temporal": dates[0] + "T00:00:00Z," + dates[-1] + "T23:59:59Z",
                    "page_size": 2000, "page_num": page,
                    "sort_key[]": ["start_date", "producer_granule_id"],
                })
                entries = data["feed"]["entry"]
                total = int(headers["CMR-Hits"])
                for entry in entries:
                    if entry["id"] in seen:
                        raise ValueError("Duplicate granule across catalog pages")
                    seen.add(entry["id"])
                count += len(entries)
                product["granules"].extend(select_required_granules(entries, set(dates)))
                if count == total:
                    product["complete"] = True
                    product["range_granules_scanned"] = count
                    break
                if not entries or count > total:
                    raise ValueError("Incomplete or inconsistent CMR pagination")
                page += 1
            print(name, len(product["granules"]), "nominal-date metadata granules", flush=True)
        except Exception as error:
            product["error_type"] = type(error).__name__
            print(name, "query incomplete:", type(error).__name__, flush=True)
        product["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / (name + ".json"), product)
        summary = build_coverage(dates, products)
        summary["expected_products"] = list(PRODUCTS)
        summary["all_expected_products_queried"] = len(products) == len(PRODUCTS)
        if len(products) != len(PRODUCTS):
            summary["all_products_metadata_dates"] = []
        summary["required_times_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        summary["retrieved_at_utc"] = product["retrieved_at_utc"]
        write_json(out / "coverage.json", summary)
    session.close()


if __name__ == "__main__":
    main()
