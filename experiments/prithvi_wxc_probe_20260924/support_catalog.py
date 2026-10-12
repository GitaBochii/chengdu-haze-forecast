"""Public CMR inventory of instant and averaged source-day dependencies."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_history_catalog import PRODUCTS, public_json, select_required_granules

BASE = Path(__file__).resolve().parent / "full_history"
PLAN = BASE / "support_aligned_plan"
AVERAGED = ("M2T1NXLND", "M2T1NXFLX", "M2T1NXRAD")


def required_dates_by_product(frame):
    instant = sorted(set(pd.to_datetime(frame["weather_time_utc"], utc=True).dt.date.astype(str)))
    average = sorted(set(pd.to_datetime(pd.concat([frame["source_left_utc"], frame["source_right_utc"]]), utc=True).dt.date.astype(str)))
    if not instant or not average:
        raise ValueError("No required weather dates")
    return {name: average if name in AVERAGED else instant for name in PRODUCTS}


def collect_pages(session, collection_id, dates, product, fetch=public_json):
    seen = set()
    total = None
    page = 1
    while True:
        data, headers = fetch(session, "granules.json", {
            "collection_concept_id": collection_id,
            "temporal": dates[0] + "T00:00:00Z," + dates[-1] + "T23:59:59Z",
            "page_size": 2000, "page_num": page,
            "sort_key[]": ["start_date", "producer_granule_id"],
        })
        hits = int(headers["CMR-Hits"])
        if total is not None and hits != total:
            raise ValueError("CMR hit count changed during pagination")
        total = hits
        entries = data["feed"]["entry"]
        for entry in entries:
            if entry["id"] in seen:
                raise ValueError("Duplicate granule across catalog pages")
            seen.add(entry["id"])
        product["granules"].extend(select_required_granules(entries, set(dates)))
        product["range_granules_scanned"] = len(seen)
        product["range_hits"] = hits
        product["pages_read"] = page
        if len(seen) == total:
            product["complete"] = True
            return
        if not entries or len(seen) > total:
            raise ValueError("Incomplete or inconsistent CMR pagination")
        page += 1


def build_coverage(required_by_product, products):
    summaries = {}
    for name in PRODUCTS:
        required = set(required_by_product[name])
        product = products.get(name, {"complete": False, "granules": []})
        found = {pd.Timestamp(g["time_start"]).tz_convert("UTC").date().isoformat() for g in product["granules"]} & required
        absent = sorted(required - found)
        complete = product["complete"]
        summaries[name] = {
            "catalog_query_started": name in products,
            "catalog_query_complete": complete,
            "source_date_rule": "source_left_utc and source_right_utc" if name in AVERAGED else "weather_time_utc",
            "required_dates": sorted(required),
            "required_date_count": len(required),
            "dates_with_metadata": len(found),
            "granule_count": len(product["granules"]),
            "missing_required_dates": absent if complete else [],
            "unresolved_required_dates": absent if not complete else [],
            "years": {year: {"required_dates": sum(d.startswith(year) for d in required),
                              "dates_with_metadata": sum(d.startswith(year) for d in found)}
                      for year in sorted({d[:4] for d in required})},
        }
    all_complete = all(p["catalog_query_complete"] for p in summaries.values())
    return {
        "dependency_scope": "Support-aligned dynamic source-day metadata; static and climatology excluded",
        "expected_products": list(PRODUCTS),
        "all_expected_products_query_complete": all_complete,
        "all_required_dates_have_metadata": all_complete and all(not p["missing_required_dates"] for p in summaries.values()),
        "data_downloaded": False,
        "preprocessing_complete": False,
        "products": summaries,
        "limitations": [
            "Catalog metadata does not prove authorization, valid numerical values or intra-day coverage.",
            "Static fields and climatology are not covered by this dynamic catalog.",
            "Reanalysis is retrospective and is not operational availability evidence.",
        ],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main():
    source = PLAN / "required_weather_times.parquet"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    required = required_dates_by_product(pd.read_parquet(source))
    old_dates = set(pd.to_datetime(pd.read_parquet(BASE / "required_weather_times.parquet")["weather_time_utc"], utc=True).dt.date.astype(str))
    out = PLAN / "catalog"
    out.mkdir(parents=True, exist_ok=True)
    products = {}
    with requests.Session() as session:
        session.trust_env = False
        session.headers["User-Agent"] = "chengdu-haze-support-catalog/1.0"
        for name in PRODUCTS:
            dates = required[name]
            product = {"short_name": name, "version": "5.12.4", "complete": False,
                       "granules": [], "required_times_sha256": source_hash,
                       "data_downloaded": False, "preprocessing_complete": False}
            products[name] = product
            try:
                data, _ = public_json(session, "collections.json", {"short_name": name, "version": "5.12.4", "page_size": 100})
                collections = [entry for entry in data["feed"]["entry"]
                               if entry.get("data_center") == "GES_DISC" and entry.get("short_name") == name
                               and entry.get("version_id") == "5.12.4"]
                if len(collections) != 1:
                    raise ValueError("Expected one exact GES DISC collection and version")
                product["collection"] = collections[0]
                product["collection_concept_id"] = collections[0]["id"]
                collect_pages(session, collections[0]["id"], dates, product)
                print(name, len(product["granules"]), "source-day metadata granules", flush=True)
            except Exception as error:
                product["error_type"] = type(error).__name__
                print(name, "query incomplete:", type(error).__name__, flush=True)
            product["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(out / (name + ".json"), product)
            report = build_coverage(required, products)
            report.update(required_times_sha256=source_hash, retrieved_at_utc=product["retrieved_at_utc"],
                          old_nominal_date_count=len(old_dates),
                          added_dates_relative_to_old_nominal={p: sorted(set(ds) - old_dates) for p, ds in required.items()})
            write_json(out / "coverage.json", report)
    if not report["all_required_dates_have_metadata"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
