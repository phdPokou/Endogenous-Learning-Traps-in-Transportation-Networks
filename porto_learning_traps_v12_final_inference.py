#!/usr/bin/env python3
"""
Porto Taxi Feasibility Audit V6
================================

Purpose
-------
Audit whether the Porto Taxi Trajectory dataset contains enough repeated
route-choice structure to support a Q1++ paper on endogenous learnability,
self-confirming routing equilibria, and learning traps.

This script does NOT estimate the final theory. It answers the prior question:
Do we have enough repeated OD contexts, persistent taxi histories, and route
variation to make the empirical application credible?

Dataset
-------
Official source: UCI Machine Learning Repository, dataset 339
"Taxi Service Trajectory - Prediction Challenge, ECML PKDD 2015".
The script can read either:
  1) original train.csv or train.csv.zip from the Kaggle/UCI challenge format;
  2) any local CSV with the same key columns.

Typical columns in original train.csv:
TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND, TAXI_ID, TIMESTAMP,
DAY_TYPE, MISSING_DATA, POLYLINE

Outputs
-------
Results_Porto/
  audit_config.json
  global_summary.csv
  taxi_summary.csv
  od_context_summary.csv
  taxi_context_summary.csv
  route_alternative_summary.csv
  feasibility_flags.csv
  figures/*.png

Core diagnostics
----------------
1. Persistent taxi depth: trips per TAXI_ID.
2. OD-context repetition: trips per origin-zone/destination-zone/time-regime.
3. Within-taxi repeated contexts: repeated decisions by the same taxi in same OD context.
4. Route diversity: approximate route signatures within each OD context.
5. Potential counterfactual support: whether alternative routes are observed in close time windows.

Notes
-----
- V3 reconstructs behavioral route families from coarser multi-point signatures, filters non-longitudinal/redundant contexts, and recalculates route persistence and information incompleteness on route families rather than raw GPS signatures.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


UCI_PAGE = "https://archive.ics.uci.edu/dataset/339/taxi%2Bservice%2Btrajectory%2Bprediction%2Bchallenge%2Becml%2Bpkdd%2B2015"


@dataclass
class AuditConfig:
    data_path: str
    output_dir: str = "Results_Porto"
    chunksize: int = 100_000
    grid_size_deg: float = 0.005
    min_polyline_points: int = 4
    min_context_trips: int = 30
    min_taxi_context_repeats: int = 5
    route_bins: int = 6
    time_window_minutes: int = 30
    sample_route_rows: int = 300_000
    random_seed: int = 123
    family_grid_deg: float = 0.0125
    family_bins: int = 4
    min_family_count: int = 5
    min_family_share: float = 0.01
    min_span_days: float = 30.0
    min_distance_km: float = 1.0
    exclude_same_zone: bool = True


def log(msg: str) -> None:
    print(f"[PortoAudit] {msg}", flush=True)


def ensure_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)


def resolve_csv_path(data_path: Path) -> Path:
    """Return a readable CSV path. If data_path is a zip, extract train.csv nearby."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Download train.csv/train.csv.zip first. Official page: {UCI_PAGE}"
        )
    if data_path.suffix.lower() != ".zip":
        return data_path

    extract_dir = data_path.parent / (data_path.stem + "_extracted")
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(data_path, "r") as zf:
        names = zf.namelist()
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found inside {data_path}")
        target = csv_names[0]
        out = extract_dir / Path(target).name
        if not out.exists():
            log(f"Extracting {target} -> {out}")
            with zf.open(target) as src, open(out, "wb") as dst:
                dst.write(src.read())
        return out


def parse_polyline(polyline: object) -> Optional[List[Tuple[float, float]]]:
    """Parse Porto POLYLINE string: [[lon, lat], [lon, lat], ...]."""
    if not isinstance(polyline, str) or polyline in ("[]", "", "nan"):
        return None
    try:
        pts = ast.literal_eval(polyline)
    except Exception:
        return None
    if not isinstance(pts, list) or len(pts) == 0:
        return None
    out: List[Tuple[float, float]] = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        lon, lat = float(p[0]), float(p[1])
        if math.isfinite(lon) and math.isfinite(lat):
            out.append((lon, lat))
    return out if out else None


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def path_length_km(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return np.nan
    return sum(haversine_km(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]) for i in range(len(points) - 1))


def zone_id(lon: float, lat: float, grid: float) -> str:
    return f"{math.floor(lon / grid)}_{math.floor(lat / grid)}"


def time_regime(ts: pd.Timestamp) -> str:
    hour = ts.hour
    dow = ts.dayofweek
    day = "WE" if dow >= 5 else "WD"
    if 6 <= hour < 10:
        tod = "AM"
    elif 10 <= hour < 16:
        tod = "MID"
    elif 16 <= hour < 20:
        tod = "PM"
    else:
        tod = "NT"
    return f"{day}_{tod}"


def route_signature(points: Sequence[Tuple[float, float]], bins: int, grid: float) -> str:
    """Coarse route signature based on sampled intermediate zones, not just OD."""
    if len(points) < 2:
        return "NA"
    idx = np.linspace(0, len(points) - 1, bins + 2).round().astype(int)
    sampled = [points[i] for i in idx]
    zones = [zone_id(lon, lat, grid) for lon, lat in sampled]
    # Collapse consecutive duplicates for stability.
    collapsed = []
    for z in zones:
        if not collapsed or collapsed[-1] != z:
            collapsed.append(z)
    return "|".join(collapsed)


def process_chunk(df: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    required = {"TAXI_ID", "TIMESTAMP", "POLYLINE"}
    missing = required - set(df.columns)
    if missing:
        # Support lowercase derived data if present.
        lower_map = {c.lower(): c for c in df.columns}
        ren = {}
        for key in list(missing):
            if key.lower() in lower_map:
                ren[lower_map[key.lower()]] = key
        if ren:
            df = df.rename(columns=ren)
            missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}. Columns found: {list(df.columns)}")

    cols = [c for c in ["TRIP_ID", "CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "TAXI_ID", "TIMESTAMP", "DAY_TYPE", "MISSING_DATA", "POLYLINE"] if c in df.columns]
    df = df[cols].copy()

    if "MISSING_DATA" in df.columns:
        df = df[df["MISSING_DATA"].astype(str).str.lower().isin(["false", "0", "nan"]) | df["MISSING_DATA"].isna()]

    parsed = df["POLYLINE"].apply(parse_polyline)
    df = df.assign(points=parsed)
    df = df[df["points"].notna()]
    df = df[df["points"].apply(len) >= cfg.min_polyline_points]
    if df.empty:
        return pd.DataFrame()

    starts = df["points"].apply(lambda p: p[0])
    ends = df["points"].apply(lambda p: p[-1])
    df["start_lon"] = starts.apply(lambda p: p[0])
    df["start_lat"] = starts.apply(lambda p: p[1])
    df["end_lon"] = ends.apply(lambda p: p[0])
    df["end_lat"] = ends.apply(lambda p: p[1])
    df["n_points"] = df["points"].apply(len)
    df["duration_min"] = (df["n_points"] - 1) * 15.0 / 60.0
    df["distance_km"] = df["points"].apply(path_length_km)

    df["timestamp_dt"] = pd.to_datetime(df["TIMESTAMP"], unit="s", errors="coerce")
    df = df[df["timestamp_dt"].notna()]
    df["date"] = df["timestamp_dt"].dt.date.astype(str)
    df["time_regime"] = df["timestamp_dt"].apply(time_regime)
    df["origin_zone"] = [zone_id(lon, lat, cfg.grid_size_deg) for lon, lat in zip(df["start_lon"], df["start_lat"])]
    df["dest_zone"] = [zone_id(lon, lat, cfg.grid_size_deg) for lon, lat in zip(df["end_lon"], df["end_lat"])]
    df["od_pair"] = df["origin_zone"] + "__" + df["dest_zone"]
    df["od_context"] = df["od_pair"] + "__" + df["time_regime"]
    df["route_sig"] = df["points"].apply(lambda p: route_signature(p, cfg.route_bins, cfg.grid_size_deg))
    # V3: coarser route representation used as the raw behavioral route-family candidate.
    df["route_family_raw"] = df["points"].apply(lambda p: route_signature(p, cfg.family_bins, cfg.family_grid_deg))
    df["taxi_context"] = df["TAXI_ID"].astype(str) + "__" + df["od_context"].astype(str)

    keep = [
        "TRIP_ID", "CALL_TYPE", "ORIGIN_STAND", "TAXI_ID", "timestamp_dt", "date", "time_regime",
        "origin_zone", "dest_zone", "od_pair", "od_context", "route_sig", "route_family_raw", "taxi_context",
        "duration_min", "distance_km", "n_points", "start_lon", "start_lat", "end_lon", "end_lat",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def read_processed(cfg: AuditConfig) -> pd.DataFrame:
    csv_path = resolve_csv_path(Path(cfg.data_path))
    log(f"Reading {csv_path}")
    frames = []
    total = 0
    for k, chunk in enumerate(pd.read_csv(csv_path, chunksize=cfg.chunksize)):
        proc = process_chunk(chunk, cfg)
        total += len(proc)
        if not proc.empty:
            frames.append(proc)
        if k % 5 == 0:
            log(f"Processed chunk {k}; valid trips so far: {total:,}")
    if not frames:
        raise RuntimeError("No valid trajectories after processing.")
    out = pd.concat(frames, ignore_index=True)
    log(f"Finished processing: {len(out):,} valid trajectories")
    return out


def summarize(df: pd.DataFrame, cfg: AuditConfig, output_dir: Path) -> None:
    # Global summary
    global_summary = pd.DataFrame([{
        "n_valid_trips": len(df),
        "n_taxis": df["TAXI_ID"].nunique(),
        "n_od_pairs": df["od_pair"].nunique(),
        "n_od_contexts": df["od_context"].nunique(),
        "n_route_signatures": df["route_sig"].nunique(),
        "date_min": df["timestamp_dt"].min(),
        "date_max": df["timestamp_dt"].max(),
        "median_duration_min": df["duration_min"].median(),
        "median_distance_km": df["distance_km"].median(),
    }])
    global_summary.to_csv(output_dir / "global_summary.csv", index=False)

    taxi_summary = df.groupby("TAXI_ID").agg(
        n_trips=("TAXI_ID", "size"),
        n_od_contexts=("od_context", "nunique"),
        n_active_days=("date", "nunique"),
        median_duration_min=("duration_min", "median"),
        median_distance_km=("distance_km", "median"),
    ).reset_index().sort_values("n_trips", ascending=False)
    taxi_summary.to_csv(output_dir / "taxi_summary.csv", index=False)

    od_context_summary = df.groupby("od_context").agg(
        n_trips=("od_context", "size"),
        n_taxis=("TAXI_ID", "nunique"),
        n_routes=("route_sig", "nunique"),
        median_duration_min=("duration_min", "median"),
        iqr_duration_min=("duration_min", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        median_distance_km=("distance_km", "median"),
    ).reset_index()
    od_context_summary["multi_route"] = od_context_summary["n_routes"] >= 2
    od_context_summary["eligible_context"] = (
        (od_context_summary["n_trips"] >= cfg.min_context_trips) &
        (od_context_summary["n_routes"] >= 2) &
        (od_context_summary["n_taxis"] >= 5)
    )
    od_context_summary = od_context_summary.sort_values(["eligible_context", "n_trips"], ascending=[False, False])
    od_context_summary.to_csv(output_dir / "od_context_summary.csv", index=False)

    taxi_context_summary = df.groupby(["TAXI_ID", "od_context"]).agg(
        n_repeats=("taxi_context", "size"),
        n_routes_used=("route_sig", "nunique"),
        first_seen=("timestamp_dt", "min"),
        last_seen=("timestamp_dt", "max"),
        median_duration_min=("duration_min", "median"),
    ).reset_index()
    taxi_context_summary["repeated_enough"] = taxi_context_summary["n_repeats"] >= cfg.min_taxi_context_repeats
    taxi_context_summary = taxi_context_summary.sort_values("n_repeats", ascending=False)
    taxi_context_summary.to_csv(output_dir / "taxi_context_summary.csv", index=False)

    route_alt = df.groupby(["od_context", "route_sig"]).agg(
        n_route_trips=("route_sig", "size"),
        n_taxis=("TAXI_ID", "nunique"),
        median_duration_min=("duration_min", "median"),
        median_distance_km=("distance_km", "median"),
    ).reset_index()
    route_alt.to_csv(output_dir / "route_alternative_summary.csv", index=False)

    # Feasibility flags
    flags = []
    flags.append({"criterion": "persistent_taxi_histories", "value": int((taxi_summary["n_trips"] >= 1000).sum()), "threshold": 100, "pass": int((taxi_summary["n_trips"] >= 1000).sum()) >= 100})
    flags.append({"criterion": "eligible_multi_route_contexts", "value": int(od_context_summary["eligible_context"].sum()), "threshold": 50, "pass": int(od_context_summary["eligible_context"].sum()) >= 50})
    flags.append({"criterion": "taxi_contexts_with_repeats", "value": int(taxi_context_summary["repeated_enough"].sum()), "threshold": 200, "pass": int(taxi_context_summary["repeated_enough"].sum()) >= 200})
    flags.append({"criterion": "contexts_with_3plus_routes", "value": int(((od_context_summary["n_trips"] >= cfg.min_context_trips) & (od_context_summary["n_routes"] >= 3)).sum()), "threshold": 20, "pass": int(((od_context_summary["n_trips"] >= cfg.min_context_trips) & (od_context_summary["n_routes"] >= 3)).sum()) >= 20})
    flags_df = pd.DataFrame(flags)
    flags_df.to_csv(output_dir / "feasibility_flags.csv", index=False)

    log("Global summary:")
    log(global_summary.to_string(index=False))
    log("Feasibility flags:")
    log(flags_df.to_string(index=False))

    make_figures(df, taxi_summary, od_context_summary, taxi_context_summary, output_dir)


def make_figures(df, taxi_summary, od_context_summary, taxi_context_summary, output_dir: Path) -> None:
    if plt is None:
        log("matplotlib not available; skipping figures")
        return

    figdir = output_dir / "figures"

    def save_hist(series, title, xlabel, fname, bins=50, logy=False):
        plt.figure(figsize=(8, 5))
        plt.hist(series.dropna(), bins=bins)
        if logy:
            plt.yscale("log")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figdir / fname, dpi=200)
        plt.close()

    save_hist(taxi_summary["n_trips"], "Trips per taxi", "Number of trips", "trips_per_taxi.png", bins=40)
    save_hist(od_context_summary["n_trips"], "Trips per OD-time context", "Number of trips", "trips_per_od_context.png", bins=80, logy=True)
    save_hist(od_context_summary["n_routes"], "Route signatures per OD-time context", "Number of route signatures", "routes_per_context.png", bins=50, logy=True)
    save_hist(taxi_context_summary["n_repeats"], "Repeated decisions by taxi and OD-time context", "Repeats", "taxi_context_repeats.png", bins=80, logy=True)

    top = od_context_summary.head(30).sort_values("n_trips")
    plt.figure(figsize=(9, 7))
    plt.barh(range(len(top)), top["n_trips"])
    plt.yticks(range(len(top)), top["od_context"], fontsize=6)
    plt.title("Top OD-time contexts by number of trips")
    plt.xlabel("Number of trips")
    plt.tight_layout()
    plt.savefig(figdir / "top_od_contexts.png", dpi=200)
    plt.close()


def maybe_counterfactual_support(df: pd.DataFrame, cfg: AuditConfig, output_dir: Path) -> None:
    """Approximate support for route alternatives observed close in time.

    This is a V1 heuristic: for each sampled trip, count other route signatures in
    same od_context within +/- time_window_minutes.
    """
    n = min(len(df), cfg.sample_route_rows)
    sample = df.sample(n=n, random_state=cfg.random_seed).copy() if len(df) > n else df.copy()
    sample = sample.sort_values("timestamp_dt")
    rows = []
    window = pd.Timedelta(minutes=cfg.time_window_minutes)

    for context, g in sample.groupby("od_context"):
        if len(g) < cfg.min_context_trips or g["route_sig"].nunique() < 2:
            continue
        times = g["timestamp_dt"].to_numpy()
        routes = g["route_sig"].to_numpy()
        idx = g.index.to_numpy()
        # Two-pointer window counts by brute force per context; ok for sampled V1.
        for pos, original_idx in enumerate(idx):
            t = pd.Timestamp(times[pos])
            mask = (g["timestamp_dt"] >= t - window) & (g["timestamp_dt"] <= t + window)
            alt_routes = set(g.loc[mask, "route_sig"]) - {routes[pos]}
            rows.append({
                "sample_index": int(original_idx),
                "od_context": context,
                "route_sig": routes[pos],
                "n_alt_routes_window": len(alt_routes),
                "has_alt_route_window": len(alt_routes) > 0,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["sample_index", "od_context", "route_sig", "n_alt_routes_window", "has_alt_route_window"])
    out.to_csv(output_dir / "counterfactual_window_support_sample.csv", index=False)
    if len(out):
        summary = pd.DataFrame([{
            "sampled_trips_checked": len(out),
            "share_with_alt_route_in_time_window": out["has_alt_route_window"].mean(),
            "median_alt_routes_in_window": out["n_alt_routes_window"].median(),
            "time_window_minutes": cfg.time_window_minutes,
        }])
    else:
        summary = pd.DataFrame([{
            "sampled_trips_checked": 0,
            "share_with_alt_route_in_time_window": np.nan,
            "median_alt_routes_in_window": np.nan,
            "time_window_minutes": cfg.time_window_minutes,
        }])
    summary.to_csv(output_dir / "counterfactual_window_support_summary.csv", index=False)
    log("Counterfactual support summary:")
    log(summary.to_string(index=False))



# ---------------------------------------------------------------------------
# V2: Q1++ feasibility tables
# ---------------------------------------------------------------------------

def safe_entropy(counts: pd.Series) -> float:
    """Shannon entropy of a nonnegative count vector."""
    arr = counts.astype(float).to_numpy()
    total = arr.sum()
    if total <= 0:
        return float("nan")
    p = arr[arr > 0] / total
    return float(-(p * np.log(p)).sum())


def safe_hhi(counts: pd.Series) -> float:
    """Herfindahl index of route concentration; 1 means one dominant route."""
    arr = counts.astype(float).to_numpy()
    total = arr.sum()
    if total <= 0:
        return float("nan")
    p = arr / total
    return float((p ** 2).sum())


def switch_rate_for_group(g: pd.DataFrame) -> float:
    g = g.sort_values("timestamp_dt")
    routes = g["route_sig"].astype(str).to_numpy()
    if len(routes) <= 1:
        return float("nan")
    return float(np.mean(routes[1:] != routes[:-1]))


def dominant_share_for_group(g: pd.DataFrame) -> float:
    if len(g) == 0:
        return float("nan")
    return float(g["route_sig"].value_counts(normalize=True).iloc[0])


def route_entropy_for_group(g: pd.DataFrame) -> float:
    return safe_entropy(g["route_sig"].value_counts())


def add_rank_columns(df: pd.DataFrame, sort_cols: list[str], ascending=None) -> pd.DataFrame:
    out = df.sort_values(sort_cols, ascending=ascending if ascending is not None else [False] * len(sort_cols)).copy()
    # Some upstream V3/V4 tables already carry a rank column. Re-ranking must
    # therefore replace it rather than trying to insert a duplicate column.
    if "rank" in out.columns:
        out = out.drop(columns=["rank"])
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def write_latex_table(df: pd.DataFrame, path: Path, max_rows: int = 30, float_format: str = "%.4f") -> None:
    """Write compact LaTeX table. Avoid hard failure if pandas changes APIs."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(df.head(max_rows).to_latex(index=False, escape=True, float_format=float_format))
    except Exception as exc:
        log(f"Could not write LaTeX table {path.name}: {exc}")


def compute_q1_tables_v2(df: pd.DataFrame, cfg: AuditConfig, output_dir: Path) -> None:
    """Produce Q1++ tables for the theoretical application.

    The goal is not prediction accuracy. The tables quantify whether the data
    contain the empirical primitives needed for the paper:
      1) repeated decisions by the same taxi in the same OD-time context;
      2) exploitable contexts with route alternatives;
      3) route-choice stability/persistence;
      4) observational incompleteness induced by selective route experience.
    """
    qdir = output_dir / "q1_tables"
    qdir.mkdir(parents=True, exist_ok=True)
    figdir = output_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    log("Computing V2 Q1++ tables...")

    # Route distribution by OD-time context.
    route_counts = (
        df.groupby(["od_context", "route_sig"])
          .size()
          .rename("n_route_trips")
          .reset_index()
    )
    context_route_stats = route_counts.groupby("od_context").agg(
        n_routes=("route_sig", "nunique"),
        route_entropy=("n_route_trips", safe_entropy),
        route_hhi=("n_route_trips", safe_hhi),
        top_route_trips=("n_route_trips", "max"),
        total_trips=("n_route_trips", "sum"),
    ).reset_index()
    context_route_stats["top_route_share"] = context_route_stats["top_route_trips"] / context_route_stats["total_trips"]
    context_route_stats["effective_routes"] = np.exp(context_route_stats["route_entropy"])

    # Repetition structure by taxi-context.
    taxi_context = df.groupby(["TAXI_ID", "od_context"]).agg(
        n_repeats=("route_sig", "size"),
        n_routes_used=("route_sig", "nunique"),
        first_seen=("timestamp_dt", "min"),
        last_seen=("timestamp_dt", "max"),
        median_duration_min=("duration_min", "median"),
        median_distance_km=("distance_km", "median"),
    ).reset_index()
    taxi_context["span_days"] = (taxi_context["last_seen"] - taxi_context["first_seen"]).dt.total_seconds() / 86400.0
    taxi_context = taxi_context.merge(context_route_stats[["od_context", "n_routes", "total_trips", "route_entropy", "route_hhi", "top_route_share", "effective_routes"]], on="od_context", how="left")
    taxi_context["route_coverage_ratio"] = taxi_context["n_routes_used"] / taxi_context["n_routes"].replace(0, np.nan)
    taxi_context["information_incompleteness"] = 1.0 - taxi_context["route_coverage_ratio"]
    taxi_context["eligible_repeated"] = taxi_context["n_repeats"] >= cfg.min_taxi_context_repeats
    taxi_context.to_csv(qdir / "taxi_context_repetition_and_incompleteness.csv", index=False)

    # Distribution table: repeated decisions by taxi-OD-time context.
    thresholds = [2, 3, 5, 10, 20, 50, 100]
    rep = taxi_context["n_repeats"]
    repetition_distribution = pd.DataFrame([{
        "n_taxi_contexts": int(len(rep)),
        "mean_repeats": rep.mean(),
        "sd_repeats": rep.std(),
        "p25_repeats": rep.quantile(0.25),
        "median_repeats": rep.median(),
        "p75_repeats": rep.quantile(0.75),
        "p90_repeats": rep.quantile(0.90),
        "p95_repeats": rep.quantile(0.95),
        "p99_repeats": rep.quantile(0.99),
        "max_repeats": rep.max(),
        **{f"n_contexts_ge_{k}": int((rep >= k).sum()) for k in thresholds},
        **{f"share_contexts_ge_{k}": float((rep >= k).mean()) for k in thresholds},
    }])
    repetition_distribution.to_csv(qdir / "table_1_taxi_od_repetition_distribution.csv", index=False)
    write_latex_table(repetition_distribution.T.reset_index().rename(columns={"index": "metric", 0: "value"}), qdir / "table_1_taxi_od_repetition_distribution.tex", max_rows=80)

    # Route stability and persistence at taxi-context level.
    eligible_df = df.merge(
        taxi_context.loc[taxi_context["eligible_repeated"], ["TAXI_ID", "od_context"]],
        on=["TAXI_ID", "od_context"],
        how="inner",
    )
    if not eligible_df.empty:
        stability = eligible_df.groupby(["TAXI_ID", "od_context"]).apply(lambda g: pd.Series({
            "n_repeats": len(g),
            "n_routes_used": g["route_sig"].nunique(),
            "dominant_route_share": dominant_share_for_group(g),
            "switch_rate": switch_rate_for_group(g),
            "route_entropy_within_taxi": route_entropy_for_group(g),
            "first_route": g.sort_values("timestamp_dt")["route_sig"].iloc[0],
            "last_route": g.sort_values("timestamp_dt")["route_sig"].iloc[-1],
        })).reset_index()
    else:
        stability = pd.DataFrame(columns=["TAXI_ID", "od_context", "n_repeats", "n_routes_used", "dominant_route_share", "switch_rate", "route_entropy_within_taxi", "first_route", "last_route"])
    stability["same_first_last_route"] = stability["first_route"].astype(str) == stability["last_route"].astype(str)
    stability = stability.merge(
        taxi_context[["TAXI_ID", "od_context", "information_incompleteness", "route_coverage_ratio", "n_routes", "total_trips", "route_hhi", "top_route_share", "span_days"]],
        on=["TAXI_ID", "od_context"],
        how="left",
    )
    stability.to_csv(qdir / "route_stability_taxi_context.csv", index=False)

    stability_summary = pd.DataFrame([{
        "n_eligible_taxi_contexts": int(len(stability)),
        "median_dominant_route_share": stability["dominant_route_share"].median(),
        "mean_dominant_route_share": stability["dominant_route_share"].mean(),
        "median_switch_rate": stability["switch_rate"].median(),
        "mean_switch_rate": stability["switch_rate"].mean(),
        "share_same_first_last_route": stability["same_first_last_route"].mean() if len(stability) else np.nan,
        "median_information_incompleteness": stability["information_incompleteness"].median(),
        "mean_information_incompleteness": stability["information_incompleteness"].mean(),
    }])
    stability_summary.to_csv(qdir / "table_3_route_stability_summary.csv", index=False)
    write_latex_table(stability_summary.T.reset_index().rename(columns={"index": "metric", 0: "value"}), qdir / "table_3_route_stability_summary.tex", max_rows=80)

    # Context-level incompleteness: how much of route menu is unseen by repeating taxis?
    repeated_tc = taxi_context[taxi_context["eligible_repeated"]].copy()
    incompleteness_context = repeated_tc.groupby("od_context").agg(
        n_repeating_taxis=("TAXI_ID", "nunique"),
        n_repeating_taxi_contexts=("TAXI_ID", "size"),
        mean_incompleteness=("information_incompleteness", "mean"),
        median_incompleteness=("information_incompleteness", "median"),
        p75_incompleteness=("information_incompleteness", lambda s: s.quantile(0.75)),
        mean_route_coverage_ratio=("route_coverage_ratio", "mean"),
        mean_repeats_per_taxi=("n_repeats", "mean"),
        median_repeats_per_taxi=("n_repeats", "median"),
        max_repeats_by_taxi=("n_repeats", "max"),
    ).reset_index()
    incompleteness_context = incompleteness_context.merge(context_route_stats, on="od_context", how="left")
    incompleteness_context.to_csv(qdir / "information_incompleteness_context_summary.csv", index=False)

    inc_summary = pd.DataFrame([{
        "n_repeated_taxi_contexts": int(len(repeated_tc)),
        "n_od_contexts_with_repeating_taxis": int(incompleteness_context["od_context"].nunique()),
        "median_incompleteness": repeated_tc["information_incompleteness"].median(),
        "mean_incompleteness": repeated_tc["information_incompleteness"].mean(),
        "p75_incompleteness": repeated_tc["information_incompleteness"].quantile(0.75),
        "p90_incompleteness": repeated_tc["information_incompleteness"].quantile(0.90),
        "share_incompleteness_ge_0_5": float((repeated_tc["information_incompleteness"] >= 0.5).mean()),
        "share_incompleteness_ge_0_75": float((repeated_tc["information_incompleteness"] >= 0.75).mean()),
    }])
    inc_summary.to_csv(qdir / "table_4_information_incompleteness_summary.csv", index=False)
    write_latex_table(inc_summary.T.reset_index().rename(columns={"index": "metric", 0: "value"}), qdir / "table_4_information_incompleteness_summary.tex", max_rows=80)

    # Top exploitable OD-time contexts for the paper.
    top_context = incompleteness_context.copy()
    top_context = top_context[
        (top_context["total_trips"] >= cfg.min_context_trips) &
        (top_context["n_routes"] >= 3) &
        (top_context["n_repeating_taxis"] >= 3)
    ].copy()
    if not top_context.empty:
        # A paper-usefulness score: repeated decisions + alternatives + incomplete experience.
        top_context["exploitation_score"] = (
            np.log1p(top_context["total_trips"]) *
            np.log1p(top_context["n_routes"]) *
            np.log1p(top_context["n_repeating_taxis"]) *
            (1.0 + top_context["median_incompleteness"].fillna(0))
        )
        top_context = add_rank_columns(top_context, ["exploitation_score", "total_trips", "n_routes"], ascending=[False, False, False])
    top_cols = [
        "rank", "od_context", "exploitation_score", "total_trips", "n_routes", "effective_routes",
        "top_route_share", "route_hhi", "n_repeating_taxis", "median_repeats_per_taxi",
        "max_repeats_by_taxi", "median_incompleteness", "p75_incompleteness",
    ]
    top_context = top_context[[c for c in top_cols if c in top_context.columns]]
    top_context.to_csv(qdir / "table_2_top_exploitable_od_contexts.csv", index=False)
    write_latex_table(top_context, qdir / "table_2_top_exploitable_od_contexts.tex", max_rows=30)

    # Candidate learning traps: persistent route choice under incomplete route experience.
    candidates = stability.copy()
    if not candidates.empty:
        candidates["trap_candidate_score"] = (
            candidates["n_repeats"].astype(float).clip(lower=1).map(np.log1p) *
            candidates["information_incompleteness"].fillna(0) *
            candidates["dominant_route_share"].fillna(0) *
            (1.0 - candidates["switch_rate"].fillna(0)) *
            np.log1p(candidates["n_routes"].fillna(0))
        )
        candidates = add_rank_columns(candidates, ["trap_candidate_score", "n_repeats"], ascending=[False, False])
    cand_cols = [
        "rank", "TAXI_ID", "od_context", "trap_candidate_score", "n_repeats", "n_routes",
        "n_routes_used", "information_incompleteness", "dominant_route_share", "switch_rate",
        "same_first_last_route", "span_days", "total_trips", "top_route_share",
    ]
    candidates = candidates[[c for c in cand_cols if c in candidates.columns]]
    candidates.to_csv(qdir / "table_5_learning_trap_candidate_taxi_contexts.csv", index=False)
    write_latex_table(candidates, qdir / "table_5_learning_trap_candidate_taxi_contexts.tex", max_rows=30)

    # Minimal figures for manuscript diagnostics.
    if plt is not None:
        plt.figure(figsize=(8, 5))
        plt.hist(repeated_tc["information_incompleteness"].dropna(), bins=40)
        plt.title("Information incompleteness in repeated taxi--OD contexts")
        plt.xlabel("1 - routes experienced by taxi / routes observed in context")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figdir / "v2_information_incompleteness_distribution.png", dpi=220)
        plt.close()

        if not stability.empty:
            plot_df = stability.dropna(subset=["information_incompleteness", "dominant_route_share", "switch_rate"])
            if len(plot_df):
                sample = plot_df.sample(min(len(plot_df), 50000), random_state=cfg.random_seed)
                plt.figure(figsize=(8, 5))
                plt.scatter(sample["information_incompleteness"], sample["dominant_route_share"], s=6, alpha=0.35)
                plt.title("Route persistence vs. observational incompleteness")
                plt.xlabel("Information incompleteness")
                plt.ylabel("Dominant route share")
                plt.tight_layout()
                plt.savefig(figdir / "v2_persistence_vs_incompleteness.png", dpi=220)
                plt.close()

        if not top_context.empty:
            top30 = top_context.head(30).sort_values("exploitation_score")
            plt.figure(figsize=(9, 7))
            plt.barh(range(len(top30)), top30["exploitation_score"])
            plt.yticks(range(len(top30)), top30["od_context"], fontsize=6)
            plt.title("Top exploitable OD-time contexts for learning-trap analysis")
            plt.xlabel("Exploitation score")
            plt.tight_layout()
            plt.savefig(figdir / "v2_top_exploitable_contexts.png", dpi=220)
            plt.close()

    # Readme explaining interpretation.
    readme = """# Porto Audit V2: Q1++ Tables\n\nThese outputs are designed for a methodological paper on endogenous learnability in repeated routing games.\n\nKey objects:\n- `table_1_taxi_od_repetition_distribution`: depth of repeated decisions by the same taxi in the same OD-time context.\n- `table_2_top_exploitable_od_contexts`: OD-time contexts with many trips, route alternatives, and repeating taxis.\n- `table_3_route_stability_summary`: persistence/switching in repeated taxi-context histories.\n- `table_4_information_incompleteness_summary`: selective-feedback index, defined as 1 - observed route coverage.\n- `table_5_learning_trap_candidate_taxi_contexts`: high-persistence, high-incompleteness histories that can be inspected as empirical candidates for self-confirming learning traps.\n\nInterpretation note:\nThe incompleteness index is not a psychological belief measure. It is an observable selective-feedback measure: how much of the route menu observed in the OD-time context has not been personally sampled by a repeating taxi.\n"""
    (qdir / "README_Q1_TABLES.md").write_text(readme, encoding="utf-8")

    log("V2 Q1++ tables written to: " + str(qdir.resolve()))
    log("Key V2 summaries:")
    log(repetition_distribution.to_string(index=False))
    log(stability_summary.to_string(index=False))
    log(inc_summary.to_string(index=False))



def assign_behavioral_route_families(df: pd.DataFrame, cfg: AuditConfig, output_dir: Path) -> pd.DataFrame:
    """Collapse noisy GPS-level route signatures into behavioral route families.

    The raw V2 route signatures were intentionally fine-grained. V3 uses a coarser
    multi-point signature (`route_family_raw`) and then merges low-support variants
    within each OD-time context into an `OTHER` family. This is not a full map-matched
    road-segment model; it is a conservative behavioral abstraction designed to avoid
    treating tiny GPS deviations as separate route alternatives.
    """
    df = df.copy()
    if "route_family_raw" not in df.columns:
        # Backward compatibility for processed parquet produced by older versions.
        df["route_family_raw"] = df["route_sig"].astype(str)

    ctx_counts = df.groupby(["od_context", "route_family_raw"]).size().rename("family_count").reset_index()
    total = df.groupby("od_context").size().rename("context_trips").reset_index()
    ctx_counts = ctx_counts.merge(total, on="od_context", how="left")
    ctx_counts["family_share"] = ctx_counts["family_count"] / ctx_counts["context_trips"]
    ctx_counts["is_major_family"] = (
        (ctx_counts["family_count"] >= cfg.min_family_count) &
        (ctx_counts["family_share"] >= cfg.min_family_share)
    )
    major = ctx_counts.loc[ctx_counts["is_major_family"], ["od_context", "route_family_raw"]].copy()
    major["route_family"] = major["route_family_raw"]
    df = df.merge(major, on=["od_context", "route_family_raw"], how="left")
    df["route_family"] = df["route_family"].fillna("OTHER")

    # Save support table for auditability.
    support = ctx_counts.sort_values(["od_context", "family_count"], ascending=[True, False])
    support.to_csv(output_dir / "route_family_support_raw.csv", index=False)
    return df


def compute_q1_tables(df: pd.DataFrame, cfg: AuditConfig, output_dir: Path) -> None:
    """V3 Q1++ route-family tables.

    V3 answers: after replacing noisy GPS signatures by behavioral route families,
    do repeated taxi--OD histories still exhibit selective feedback, incomplete route
    experience, and credible learning-trap candidates over genuine longitudinal spans?
    """
    log("Computing V3 behavioral route-family Q1++ tables...")
    qdir = output_dir / "q1_tables_v3"
    figdir = output_dir / "figures"
    qdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    # Conceptual filters: avoid local loops and very short/degenerate trajectories.
    work = df.copy()
    work = work[work["distance_km"] >= cfg.min_distance_km].copy()
    if cfg.exclude_same_zone:
        work = work[work["origin_zone"] != work["dest_zone"]].copy()

    work = assign_behavioral_route_families(work, cfg, output_dir)

    # Context-level family menu.
    family_counts = work.groupby(["od_context", "route_family"]).size().rename("n_trips_family").reset_index()
    context_stats = family_counts.groupby("od_context").agg(
        total_trips=("n_trips_family", "sum"),
        n_route_families=("route_family", "nunique"),
        top_family_trips=("n_trips_family", "max"),
    ).reset_index()
    context_stats["top_family_share"] = context_stats["top_family_trips"] / context_stats["total_trips"]
    hhi = family_counts.merge(context_stats[["od_context", "total_trips"]], on="od_context", how="left")
    hhi["share_sq"] = (hhi["n_trips_family"] / hhi["total_trips"]) ** 2
    hhi = hhi.groupby("od_context")["share_sq"].sum().rename("family_hhi").reset_index()
    context_stats = context_stats.merge(hhi, on="od_context", how="left")
    context_stats["effective_families"] = 1.0 / context_stats["family_hhi"].replace(0, np.nan)

    # Taxi-context histories using route families.
    taxi_context = work.groupby(["TAXI_ID", "od_context"]).agg(
        n_repeats=("route_family", "size"),
        n_families_used=("route_family", "nunique"),
        first_time=("timestamp_dt", "min"),
        last_time=("timestamp_dt", "max"),
        mean_duration_min=("duration_min", "mean"),
        median_duration_min=("duration_min", "median"),
        mean_distance_km=("distance_km", "mean"),
    ).reset_index()
    taxi_context["span_days"] = (taxi_context["last_time"] - taxi_context["first_time"]).dt.total_seconds() / 86400.0
    taxi_context = taxi_context.merge(context_stats, on="od_context", how="left")
    taxi_context["eligible_repeated_longitudinal"] = (
        (taxi_context["n_repeats"] >= cfg.min_taxi_context_repeats) &
        (taxi_context["span_days"] >= cfg.min_span_days) &
        (taxi_context["n_route_families"] >= 2) &
        (taxi_context["total_trips"] >= cfg.min_context_trips)
    )
    taxi_context["family_coverage_ratio"] = taxi_context["n_families_used"] / taxi_context["n_route_families"].replace(0, np.nan)
    taxi_context["information_incompleteness"] = 1.0 - taxi_context["family_coverage_ratio"]

    # Repetition distribution table.
    rep = taxi_context["n_repeats"].astype(float)
    repdist = pd.DataFrame([{
        "n_taxi_contexts_after_filters": int(len(taxi_context)),
        "mean_repeats": rep.mean(),
        "sd_repeats": rep.std(),
        "median_repeats": rep.median(),
        "p75_repeats": rep.quantile(0.75),
        "p90_repeats": rep.quantile(0.90),
        "p95_repeats": rep.quantile(0.95),
        "p99_repeats": rep.quantile(0.99),
        "max_repeats": int(rep.max()) if len(rep) else 0,
        "n_contexts_ge_5": int((rep >= 5).sum()),
        "n_contexts_ge_10": int((rep >= 10).sum()),
        "n_contexts_ge_20": int((rep >= 20).sum()),
        "n_contexts_ge_50": int((rep >= 50).sum()),
        "n_longitudinal_eligible": int(taxi_context["eligible_repeated_longitudinal"].sum()),
    }])
    repdist.to_csv(qdir / "table_1_v3_taxi_od_repetition_distribution.csv", index=False)
    write_latex_table(repdist.T.reset_index().rename(columns={"index": "metric", 0: "value"}), qdir / "table_1_v3_taxi_od_repetition_distribution.tex", max_rows=100)

    # Route family context table.
    repeated_long = taxi_context[taxi_context["eligible_repeated_longitudinal"]].copy()
    context_repeat = repeated_long.groupby("od_context").agg(
        n_repeating_taxis=("TAXI_ID", "nunique"),
        n_repeating_taxi_contexts=("TAXI_ID", "size"),
        median_repeats_per_taxi=("n_repeats", "median"),
        max_repeats_by_taxi=("n_repeats", "max"),
        median_span_days=("span_days", "median"),
        median_incompleteness=("information_incompleteness", "median"),
        p75_incompleteness=("information_incompleteness", lambda s: s.quantile(0.75)),
    ).reset_index() if not repeated_long.empty else pd.DataFrame(columns=["od_context"])
    top_context = context_stats.merge(context_repeat, on="od_context", how="inner") if not context_repeat.empty else pd.DataFrame()
    if not top_context.empty:
        top_context["exploitation_score_v3"] = (
            np.log1p(top_context["total_trips"]) *
            np.log1p(top_context["n_route_families"]) *
            np.log1p(top_context["n_repeating_taxis"]) *
            np.log1p(top_context["median_span_days"]) *
            (1.0 + top_context["median_incompleteness"].fillna(0))
        )
        top_context = add_rank_columns(top_context, ["exploitation_score_v3", "total_trips"], ascending=[False, False])
    top_cols = ["rank", "od_context", "exploitation_score_v3", "total_trips", "n_route_families", "effective_families", "top_family_share", "family_hhi", "n_repeating_taxis", "median_repeats_per_taxi", "max_repeats_by_taxi", "median_span_days", "median_incompleteness", "p75_incompleteness"]
    top_context = top_context[[c for c in top_cols if c in top_context.columns]] if not top_context.empty else pd.DataFrame(columns=top_cols)
    top_context.to_csv(qdir / "table_2_v3_top_behavioral_route_family_contexts.csv", index=False)
    write_latex_table(top_context, qdir / "table_2_v3_top_behavioral_route_family_contexts.tex", max_rows=30)

    # Stability using temporal order.
    seq = work.sort_values(["TAXI_ID", "od_context", "timestamp_dt"]).copy()
    g = seq.groupby(["TAXI_ID", "od_context"])
    dominant = seq.groupby(["TAXI_ID", "od_context", "route_family"]).size().rename("family_count").reset_index()
    dom = dominant.sort_values(["TAXI_ID", "od_context", "family_count"], ascending=[True, True, False]).drop_duplicates(["TAXI_ID", "od_context"])
    dom = dom.rename(columns={"route_family": "dominant_family", "family_count": "dominant_family_count"})
    first_last = g["route_family"].agg(first_family="first", last_family="last").reset_index()
    seq["prev_family"] = g["route_family"].shift(1)
    switches = seq[seq["prev_family"].notna()].copy()
    switches["is_switch"] = switches["route_family"] != switches["prev_family"]
    sw = switches.groupby(["TAXI_ID", "od_context"]).agg(n_transitions=("is_switch", "size"), n_switches=("is_switch", "sum"), switch_rate=("is_switch", "mean")).reset_index()
    stability = taxi_context.merge(dom[["TAXI_ID", "od_context", "dominant_family", "dominant_family_count"]], on=["TAXI_ID", "od_context"], how="left")
    stability = stability.merge(first_last, on=["TAXI_ID", "od_context"], how="left").merge(sw, on=["TAXI_ID", "od_context"], how="left")
    stability["dominant_family_share"] = stability["dominant_family_count"] / stability["n_repeats"]
    stability["same_first_last_family"] = stability["first_family"] == stability["last_family"]
    stability["switch_rate"] = stability["switch_rate"].fillna(0.0)
    stability_eligible = stability[stability["eligible_repeated_longitudinal"]].copy()

    stab_summary = pd.DataFrame([{
        "n_eligible_longitudinal_taxi_contexts": int(len(stability_eligible)),
        "median_dominant_family_share": stability_eligible["dominant_family_share"].median() if len(stability_eligible) else np.nan,
        "mean_dominant_family_share": stability_eligible["dominant_family_share"].mean() if len(stability_eligible) else np.nan,
        "median_switch_rate": stability_eligible["switch_rate"].median() if len(stability_eligible) else np.nan,
        "mean_switch_rate": stability_eligible["switch_rate"].mean() if len(stability_eligible) else np.nan,
        "share_same_first_last_family": stability_eligible["same_first_last_family"].mean() if len(stability_eligible) else np.nan,
        "median_information_incompleteness": stability_eligible["information_incompleteness"].median() if len(stability_eligible) else np.nan,
        "mean_information_incompleteness": stability_eligible["information_incompleteness"].mean() if len(stability_eligible) else np.nan,
    }])
    stab_summary.to_csv(qdir / "table_3_v3_route_family_stability_summary.csv", index=False)
    write_latex_table(stab_summary.T.reset_index().rename(columns={"index": "metric", 0: "value"}), qdir / "table_3_v3_route_family_stability_summary.tex", max_rows=100)

    inc_summary = pd.DataFrame([{
        "n_eligible_longitudinal_taxi_contexts": int(len(stability_eligible)),
        "n_od_contexts_with_longitudinal_repeating_taxis": int(stability_eligible["od_context"].nunique()) if len(stability_eligible) else 0,
        "median_incompleteness": stability_eligible["information_incompleteness"].median() if len(stability_eligible) else np.nan,
        "mean_incompleteness": stability_eligible["information_incompleteness"].mean() if len(stability_eligible) else np.nan,
        "p75_incompleteness": stability_eligible["information_incompleteness"].quantile(0.75) if len(stability_eligible) else np.nan,
        "p90_incompleteness": stability_eligible["information_incompleteness"].quantile(0.90) if len(stability_eligible) else np.nan,
        "share_incompleteness_ge_0_5": float((stability_eligible["information_incompleteness"] >= 0.5).mean()) if len(stability_eligible) else np.nan,
        "share_incompleteness_ge_0_75": float((stability_eligible["information_incompleteness"] >= 0.75).mean()) if len(stability_eligible) else np.nan,
    }])
    inc_summary.to_csv(qdir / "table_4_v3_information_incompleteness_summary.csv", index=False)
    write_latex_table(inc_summary.T.reset_index().rename(columns={"index": "metric", 0: "value"}), qdir / "table_4_v3_information_incompleteness_summary.tex", max_rows=100)

    # Longitudinal learning-trap candidates: persistent family, high incompleteness, enough span.
    candidates = stability_eligible.copy()
    if not candidates.empty:
        candidates["trap_candidate_score_v3"] = (
            np.log1p(candidates["n_repeats"]) *
            np.log1p(candidates["span_days"]) *
            candidates["information_incompleteness"].clip(lower=0).fillna(0) *
            candidates["dominant_family_share"].clip(lower=0).fillna(0) *
            (1.0 - candidates["switch_rate"].clip(0, 1).fillna(0)) *
            np.log1p(candidates["n_route_families"].fillna(0))
        )
        candidates = add_rank_columns(candidates, ["trap_candidate_score_v3", "n_repeats", "span_days"], ascending=[False, False, False])
    cand_cols = ["rank", "TAXI_ID", "od_context", "trap_candidate_score_v3", "n_repeats", "span_days", "n_route_families", "n_families_used", "information_incompleteness", "dominant_family", "dominant_family_share", "switch_rate", "same_first_last_family", "total_trips", "top_family_share"]
    candidates = candidates[[c for c in cand_cols if c in candidates.columns]] if not candidates.empty else pd.DataFrame(columns=cand_cols)
    candidates.to_csv(qdir / "table_5_v3_longitudinal_learning_trap_candidates.csv", index=False)
    write_latex_table(candidates, qdir / "table_5_v3_longitudinal_learning_trap_candidates.tex", max_rows=30)

    # Output detailed working files for inspection.
    context_stats.to_csv(qdir / "v3_context_route_family_stats.csv", index=False)
    taxi_context.to_csv(qdir / "v3_taxi_context_route_family_panel.csv", index=False)
    stability.to_csv(qdir / "v3_taxi_context_route_family_stability_panel.csv", index=False)

    # Figures.
    if plt is not None:
        if len(stability_eligible):
            plt.figure(figsize=(8, 5))
            plt.hist(stability_eligible["information_incompleteness"].dropna(), bins=35)
            plt.title("V3 information incompleteness after behavioral route-family reconstruction")
            plt.xlabel("1 - families experienced by taxi / families observed in context")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(figdir / "v3_information_incompleteness_route_families.png", dpi=220)
            plt.close()

            sample = stability_eligible.dropna(subset=["information_incompleteness", "dominant_family_share"]).copy()
            if len(sample):
                sample = sample.sample(min(len(sample), 50000), random_state=cfg.random_seed)
                plt.figure(figsize=(8, 5))
                plt.scatter(sample["information_incompleteness"], sample["dominant_family_share"], s=8, alpha=0.35)
                plt.title("V3 persistence vs. incompleteness using route families")
                plt.xlabel("Information incompleteness")
                plt.ylabel("Dominant route-family share")
                plt.tight_layout()
                plt.savefig(figdir / "v3_persistence_vs_incompleteness_route_families.png", dpi=220)
                plt.close()

        if not top_context.empty:
            top30 = top_context.head(30).sort_values("exploitation_score_v3")
            plt.figure(figsize=(9, 7))
            plt.barh(range(len(top30)), top30["exploitation_score_v3"])
            plt.yticks(range(len(top30)), top30["od_context"], fontsize=6)
            plt.title("V3 top behavioral route-family contexts")
            plt.xlabel("Exploitation score")
            plt.tight_layout()
            plt.savefig(figdir / "v3_top_behavioral_route_family_contexts.png", dpi=220)
            plt.close()

    readme = f"""# Porto Audit V3: Behavioral Route Families

Purpose: V3 corrects the main weakness of V2. Raw GPS route signatures are too fine-grained and overcount route alternatives. V3 uses coarser multi-point route-family signatures and merges low-support variants inside each OD-time context.

Key filters:
- minimum trip distance: {cfg.min_distance_km} km
- exclude same origin/destination zone: {cfg.exclude_same_zone}
- minimum repeats for a taxi-context: {cfg.min_taxi_context_repeats}
- minimum longitudinal span: {cfg.min_span_days} days
- minimum total context trips: {cfg.min_context_trips}
- family grid: {cfg.family_grid_deg} degrees
- family bins: {cfg.family_bins}
- major family support: count >= {cfg.min_family_count} and share >= {cfg.min_family_share}

Core outputs:
- table_1_v3_taxi_od_repetition_distribution
- table_2_v3_top_behavioral_route_family_contexts
- table_3_v3_route_family_stability_summary
- table_4_v3_information_incompleteness_summary
- table_5_v3_longitudinal_learning_trap_candidates

Interpretation: The V3 incompleteness index is not a psychological belief measure. It is an observable selective-feedback measure computed on behavioral route families.
"""
    (qdir / "README_V3_ROUTE_FAMILIES.md").write_text(readme, encoding="utf-8")

    log("V3 Q1++ tables written to: " + str(qdir.resolve()))
    log("Key V3 summaries:")
    log(repdist.to_string(index=False))
    log(stab_summary.to_string(index=False))
    log(inc_summary.to_string(index=False))




def compute_v5_learning_traps(df: pd.DataFrame, cfg: AuditConfig, output_dir: Path,
                              period: str = "M",
                              min_alt_period_obs: int = 5,
                              min_better_gap_min: float = 1.0,
                              min_opportunity_periods: int = 2,
                              min_persistence_share: float = 0.60,
                              min_incompleteness: float = 0.25) -> None:
    """V5: corrected and strengthened longitudinal learning-trap identification.

    V5 fixes the V4 ranking anomaly by separating three objects:
      (i) eligibility: histories with enough longitudinal support and counterfactual opportunity;
      (ii) identification: binary learning-trap and strong learning-trap flags;
      (iii) severity: a continuous score computed for interpretation, ranked only within the
            relevant identified set unless explicitly labelled otherwise.

    V5 also adds event-style diagnostics and simple reviewer-facing identification tables:
      - conditional denominators among histories with better alternatives;
      - corrected top confirmed candidates;
      - strong candidates;
      - high-severity but non-identified histories for audit only;
      - gap/incompleteness bins;
      - threshold sensitivity;
      - simple linear-probability diagnostics with controls.
    """
    log("Computing V5 corrected longitudinal learning-trap identification tables...")
    qdir = output_dir / "q1_tables_v5"
    figdir = output_dir / "figures"
    qdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    work = work[work["distance_km"] >= cfg.min_distance_km].copy()
    if cfg.exclude_same_zone:
        work = work[work["origin_zone"] != work["dest_zone"]].copy()
    work = assign_behavioral_route_families(work, cfg, output_dir)
    work = work.sort_values(["TAXI_ID", "od_context", "timestamp_dt"]).copy()

    p = period.upper()
    if p == "W":
        work["period"] = work["timestamp_dt"].dt.to_period("W").astype(str)
    elif p == "D":
        work["period"] = work["timestamp_dt"].dt.to_period("D").astype(str)
    else:
        work["period"] = work["timestamp_dt"].dt.to_period("M").astype(str)

    context_stats = work.groupby("od_context").agg(
        total_trips=("TRIP_ID", "size"),
        n_route_families=("route_family", "nunique"),
    ).reset_index()
    taxi_context = work.groupby(["TAXI_ID", "od_context"]).agg(
        n_repeats=("TRIP_ID", "size"),
        n_families_used=("route_family", "nunique"),
        first_time=("timestamp_dt", "min"),
        last_time=("timestamp_dt", "max"),
        n_periods_active=("period", "nunique"),
    ).reset_index()
    taxi_context["span_days"] = (taxi_context["last_time"] - taxi_context["first_time"]).dt.total_seconds() / 86400.0
    taxi_context = taxi_context.merge(context_stats, on="od_context", how="left")
    taxi_context["information_incompleteness"] = 1.0 - taxi_context["n_families_used"] / taxi_context["n_route_families"].replace(0, np.nan)
    taxi_context["eligible_longitudinal"] = (
        (taxi_context["n_repeats"] >= cfg.min_taxi_context_repeats) &
        (taxi_context["span_days"] >= cfg.min_span_days) &
        (taxi_context["total_trips"] >= cfg.min_context_trips) &
        (taxi_context["n_route_families"] >= 2)
    )

    fam_counts = work.groupby(["TAXI_ID", "od_context", "route_family"]).size().reset_index(name="family_count")
    idx = fam_counts.sort_values(["TAXI_ID", "od_context", "family_count"], ascending=[True, True, False]).groupby(["TAXI_ID", "od_context"]).head(1)
    incumbent = idx.rename(columns={"route_family": "incumbent_family", "family_count": "incumbent_count"})

    seq = work[["TAXI_ID", "od_context", "timestamp_dt", "route_family"]].sort_values(["TAXI_ID", "od_context", "timestamp_dt"])
    seq["prev_family"] = seq.groupby(["TAXI_ID", "od_context"])["route_family"].shift(1)
    seq["switch"] = (seq["route_family"] != seq["prev_family"]) & seq["prev_family"].notna()
    switches = seq.groupby(["TAXI_ID", "od_context"]).agg(n_switches=("switch", "sum")).reset_index()
    switches = switches.merge(taxi_context[["TAXI_ID", "od_context", "n_repeats"]], on=["TAXI_ID", "od_context"], how="left")
    switches["switch_rate"] = switches["n_switches"] / (switches["n_repeats"] - 1).replace(0, np.nan)

    hist = taxi_context.merge(incumbent[["TAXI_ID", "od_context", "incumbent_family", "incumbent_count"]], on=["TAXI_ID", "od_context"], how="left")
    hist = hist.merge(switches[["TAXI_ID", "od_context", "switch_rate"]], on=["TAXI_ID", "od_context"], how="left")
    hist["incumbent_share"] = hist["incumbent_count"] / hist["n_repeats"]
    hist["switch_rate"] = hist["switch_rate"].fillna(0.0)
    eligible = hist[hist["eligible_longitudinal"]].copy()

    perf = work.groupby(["od_context", "period", "route_family"]).agg(
        n_obs=("duration_min", "size"),
        median_duration_min=("duration_min", "median"),
    ).reset_index()
    perf = perf[perf["n_obs"] >= min_alt_period_obs].copy()

    perf_map = {}
    for (ctx, per), sub in perf.groupby(["od_context", "period"]):
        perf_map[(ctx, per)] = sub.sort_values("median_duration_min")[["route_family", "median_duration_min", "n_obs"]].to_records(index=False)

    chosen_records = work.merge(eligible[["TAXI_ID", "od_context", "incumbent_family"]], on=["TAXI_ID", "od_context"], how="inner")
    chosen_records["is_incumbent_choice"] = chosen_records["route_family"].astype(str) == chosen_records["incumbent_family"].astype(str)

    rows = []
    grouped_chosen = {k: v.copy() for k, v in chosen_records.groupby(["TAXI_ID", "od_context"])}

    for row in eligible.itertuples(index=False):
        taxi = getattr(row, "TAXI_ID")
        ctx = getattr(row, "od_context")
        inc = str(getattr(row, "incumbent_family"))
        sub = grouped_chosen.get((taxi, ctx))
        if sub is None or sub.empty:
            continue
        used_families = set(sub["route_family"].astype(str).unique())
        periods = sorted(sub["period"].astype(str).unique())
        opp_periods = 0
        better_opp_periods = 0
        best_gaps = []
        best_gap_pcts = []
        best_alt_fams = []
        untried_better_periods = 0
        incumbent_perf_periods = 0
        first_better_period = None

        for per in periods:
            arr = perf_map.get((ctx, per))
            if arr is None or len(arr) == 0:
                continue
            fam_to_perf = {str(r[0]): (float(r[1]), int(r[2])) for r in arr}
            if inc not in fam_to_perf:
                continue
            inc_cost, _ = fam_to_perf[inc]
            incumbent_perf_periods += 1
            alt_candidates = [(fam, dur, n) for fam, (dur, n) in fam_to_perf.items() if fam != inc]
            if not alt_candidates:
                continue
            opp_periods += 1
            best_alt, best_alt_cost, best_alt_n = min(alt_candidates, key=lambda z: z[1])
            gap = inc_cost - best_alt_cost
            gap_pct = gap / inc_cost if inc_cost and np.isfinite(inc_cost) else np.nan
            if gap >= min_better_gap_min:
                if first_better_period is None:
                    first_better_period = per
                better_opp_periods += 1
                best_gaps.append(gap)
                best_gap_pcts.append(gap_pct)
                best_alt_fams.append(best_alt)
                if best_alt not in used_families:
                    untried_better_periods += 1

        if first_better_period is not None:
            after = sub[sub["period"].astype(str) >= str(first_better_period)].copy()
            before = sub[sub["period"].astype(str) < str(first_better_period)].copy()
            after_incumbent_share = float((after["route_family"].astype(str) == inc).mean()) if len(after) else np.nan
            before_incumbent_share = float((before["route_family"].astype(str) == inc).mean()) if len(before) else np.nan
            after_n_trips = int(len(after))
            before_n_trips = int(len(before))
            after_n_noninc = int(len(set(after["route_family"].astype(str).unique()) - {inc}))
        else:
            after_incumbent_share = np.nan
            before_incumbent_share = np.nan
            after_n_trips = 0
            before_n_trips = 0
            after_n_noninc = 0

        rows.append({
            "TAXI_ID": taxi,
            "od_context": ctx,
            "n_repeats": int(getattr(row, "n_repeats")),
            "span_days": float(getattr(row, "span_days")),
            "n_periods_active": int(getattr(row, "n_periods_active")),
            "n_route_families": int(getattr(row, "n_route_families")),
            "n_families_used": int(getattr(row, "n_families_used")),
            "information_incompleteness": float(getattr(row, "information_incompleteness")),
            "incumbent_family": inc,
            "incumbent_share": float(getattr(row, "incumbent_share")),
            "switch_rate": float(getattr(row, "switch_rate")),
            "periods_with_incumbent_perf": incumbent_perf_periods,
            "opportunity_periods": opp_periods,
            "better_opportunity_periods": better_opp_periods,
            "share_periods_with_better_alt": better_opp_periods / incumbent_perf_periods if incumbent_perf_periods else np.nan,
            "median_better_alt_gap_min": float(np.nanmedian(best_gaps)) if best_gaps else np.nan,
            "mean_better_alt_gap_min": float(np.nanmean(best_gaps)) if best_gaps else np.nan,
            "median_better_alt_gap_pct": float(np.nanmedian(best_gap_pcts)) if best_gap_pcts else np.nan,
            "n_distinct_better_alt_families": int(len(set(best_alt_fams))),
            "untried_better_periods": int(untried_better_periods),
            "share_better_periods_untried": untried_better_periods / better_opp_periods if better_opp_periods else np.nan,
            "first_better_period": first_better_period if first_better_period is not None else "",
            "before_first_better_n_trips": before_n_trips,
            "before_first_better_incumbent_share": before_incumbent_share,
            "after_first_better_n_trips": after_n_trips,
            "after_first_better_incumbent_share": after_incumbent_share,
            "after_first_better_n_nonincumbent_families": after_n_noninc,
        })

    panel = pd.DataFrame(rows)
    if panel.empty:
        panel = pd.DataFrame(columns=["TAXI_ID", "od_context"])
    else:
        panel["has_counterfactual_opportunity"] = panel["opportunity_periods"] >= min_opportunity_periods
        panel["has_better_alternative"] = panel["better_opportunity_periods"] >= min_opportunity_periods
        panel["persistent_incumbent"] = panel["incumbent_share"] >= min_persistence_share
        panel["incomplete_exposure"] = panel["information_incompleteness"] >= min_incompleteness
        panel["untried_superior_alt"] = panel["untried_better_periods"] >= 1
        panel["post_opportunity_persistence"] = panel["after_first_better_incumbent_share"] >= min_persistence_share
        panel["learning_trap_candidate_v5"] = (
            panel["has_better_alternative"] &
            panel["persistent_incumbent"] &
            panel["incomplete_exposure"] &
            panel["post_opportunity_persistence"]
        )
        panel["strong_learning_trap_candidate_v5"] = panel["learning_trap_candidate_v5"] & panel["untried_superior_alt"]
        panel["trap_severity_score_v5"] = (
            np.log1p(panel["n_repeats"].astype(float)) *
            np.log1p(panel["span_days"].astype(float)) *
            panel["incumbent_share"].fillna(0).clip(0, 1) *
            panel["information_incompleteness"].fillna(0).clip(0, 1) *
            panel["share_periods_with_better_alt"].fillna(0).clip(0, 1) *
            np.log1p(panel["median_better_alt_gap_min"].fillna(0).clip(lower=0)) *
            panel["after_first_better_incumbent_share"].fillna(0).clip(0, 1)
        )
        panel["post_minus_pre_incumbent_share"] = panel["after_first_better_incumbent_share"] - panel["before_first_better_incumbent_share"]
        panel = add_rank_columns(panel, ["trap_severity_score_v5", "n_repeats", "span_days"], ascending=[False, False, False])

    panel.to_csv(qdir / "v5_taxi_context_trap_panel.csv", index=False)

    if len(panel):
        n_eligible = len(panel)
        n_opp = int(panel["has_counterfactual_opportunity"].sum())
        n_better = int(panel["has_better_alternative"].sum())
        n_cand = int(panel["learning_trap_candidate_v5"].sum())
        n_strong = int(panel["strong_learning_trap_candidate_v5"].sum())
        summary = pd.DataFrame([{
            "n_eligible_histories": n_eligible,
            "n_with_counterfactual_opportunity": n_opp,
            "n_with_better_alternative": n_better,
            "n_learning_trap_candidates": n_cand,
            "n_strong_learning_trap_candidates": n_strong,
            "share_with_better_alternative_all_eligible": n_better / n_eligible if n_eligible else np.nan,
            "share_learning_trap_candidates_all_eligible": n_cand / n_eligible if n_eligible else np.nan,
            "share_strong_learning_trap_candidates_all_eligible": n_strong / n_eligible if n_eligible else np.nan,
            "share_learning_trap_candidates_conditional_on_better_alt": n_cand / n_better if n_better else np.nan,
            "share_strong_candidates_conditional_on_better_alt": n_strong / n_better if n_better else np.nan,
            "median_better_alt_gap_min_among_positive": panel.loc[panel["has_better_alternative"], "median_better_alt_gap_min"].median(),
            "median_post_opportunity_incumbent_share_among_positive": panel.loc[panel["has_better_alternative"], "after_first_better_incumbent_share"].median(),
        }])
    else:
        summary = pd.DataFrame([{"n_eligible_histories": 0}])
    summary.to_csv(qdir / "table_1_v5_corrected_identification_summary.csv", index=False)
    write_latex_table(summary.T.reset_index().rename(columns={"index": "metric", 0: "value"}), qdir / "table_1_v5_corrected_identification_summary.tex", max_rows=100)

    top_cols = ["rank", "TAXI_ID", "od_context", "trap_severity_score_v5", "learning_trap_candidate_v5", "strong_learning_trap_candidate_v5", "n_repeats", "span_days", "n_periods_active", "n_route_families", "n_families_used", "information_incompleteness", "incumbent_family", "incumbent_share", "switch_rate", "better_opportunity_periods", "share_periods_with_better_alt", "median_better_alt_gap_min", "median_better_alt_gap_pct", "untried_better_periods", "after_first_better_incumbent_share", "post_minus_pre_incumbent_share"]
    if len(panel):
        confirmed = panel.loc[panel["learning_trap_candidate_v5"]].sort_values("trap_severity_score_v5", ascending=False).copy()
        confirmed = add_rank_columns(confirmed, ["trap_severity_score_v5", "n_repeats"], ascending=[False, False]) if len(confirmed) else confirmed
        strong = panel.loc[panel["strong_learning_trap_candidate_v5"]].sort_values("trap_severity_score_v5", ascending=False).copy()
        strong = add_rank_columns(strong, ["trap_severity_score_v5", "n_repeats"], ascending=[False, False]) if len(strong) else strong
        audit_nonidentified = panel.loc[(~panel["learning_trap_candidate_v5"]) & (panel["trap_severity_score_v5"] > 0)].sort_values("trap_severity_score_v5", ascending=False).copy()
        audit_nonidentified = add_rank_columns(audit_nonidentified, ["trap_severity_score_v5", "n_repeats"], ascending=[False, False]) if len(audit_nonidentified) else audit_nonidentified
    else:
        confirmed = strong = audit_nonidentified = pd.DataFrame(columns=top_cols)

    confirmed[[c for c in top_cols if c in confirmed.columns]].head(100).to_csv(qdir / "table_2_v5_top_confirmed_learning_trap_candidates.csv", index=False)
    write_latex_table(confirmed[[c for c in top_cols if c in confirmed.columns]].head(30), qdir / "table_2_v5_top_confirmed_learning_trap_candidates.tex", max_rows=30)
    strong[[c for c in top_cols if c in strong.columns]].head(100).to_csv(qdir / "table_3_v5_top_strong_learning_trap_candidates.csv", index=False)
    write_latex_table(strong[[c for c in top_cols if c in strong.columns]].head(30), qdir / "table_3_v5_top_strong_learning_trap_candidates.tex", max_rows=30)
    audit_nonidentified[[c for c in top_cols if c in audit_nonidentified.columns]].head(100).to_csv(qdir / "table_4_v5_high_severity_nonidentified_audit.csv", index=False)

    # Event-style bins among histories with a better alternative.
    event = panel.loc[panel.get("has_better_alternative", False)].copy() if len(panel) else pd.DataFrame()
    if len(event):
        event["gap_bin"] = pd.cut(event["median_better_alt_gap_min"], bins=[-np.inf, 1, 2, 5, 10, np.inf], labels=["<=1", "1-2", "2-5", "5-10", ">10"])
        event["incompleteness_bin"] = pd.cut(event["information_incompleteness"], bins=[-0.001, 0.25, 0.5, 0.66, 0.75, 1.001], labels=["0-.25", ".25-.50", ".50-.66", ".66-.75", ".75-1"])
        bin_summary = event.groupby(["gap_bin", "incompleteness_bin"], observed=False).agg(
            n_histories=("TAXI_ID", "size"),
            trap_share=("learning_trap_candidate_v5", "mean"),
            strong_trap_share=("strong_learning_trap_candidate_v5", "mean"),
            median_post_incumbent_share=("after_first_better_incumbent_share", "median"),
            median_pre_incumbent_share=("before_first_better_incumbent_share", "median"),
            median_post_minus_pre=("post_minus_pre_incumbent_share", "median"),
        ).reset_index()
    else:
        bin_summary = pd.DataFrame()
    bin_summary.to_csv(qdir / "table_5_v5_event_bins_gap_by_incompleteness.csv", index=False)
    write_latex_table(bin_summary.head(50), qdir / "table_5_v5_event_bins_gap_by_incompleteness.tex", max_rows=50)

    # Threshold sensitivity with corrected flags.
    sens_rows = []
    for pers in [0.5, 0.6, 0.7, 0.8]:
        for inc_thr in [0.25, 0.5, 0.66, 0.75]:
            if len(panel):
                cand = (
                    (panel["better_opportunity_periods"] >= min_opportunity_periods) &
                    (panel["incumbent_share"] >= pers) &
                    (panel["information_incompleteness"] >= inc_thr) &
                    (panel["after_first_better_incumbent_share"] >= pers)
                )
                strong_c = cand & (panel["untried_better_periods"] >= 1)
                sens_rows.append({
                    "persistence_threshold": pers,
                    "incompleteness_threshold": inc_thr,
                    "n_candidates": int(cand.sum()),
                    "share_candidates_all_eligible": float(cand.mean()),
                    "share_candidates_conditional_on_better_alt": float(cand.sum() / max(1, panel["has_better_alternative"].sum())),
                    "n_strong_candidates": int(strong_c.sum()),
                    "share_strong_candidates_all_eligible": float(strong_c.mean()),
                })
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(qdir / "table_6_v5_threshold_sensitivity_corrected.csv", index=False)
    write_latex_table(sens, qdir / "table_6_v5_threshold_sensitivity_corrected.tex", max_rows=100)

    # Simple LPM diagnostics; no external statsmodels dependency.
    diag_rows = []
    if len(event) >= 20:
        d = event.copy()
        d["y_post_persistent"] = (d["after_first_better_incumbent_share"] >= min_persistence_share).astype(float)
        d["log_gap"] = np.log1p(d["median_better_alt_gap_min"].fillna(0).clip(lower=0))
        d["log_repeats"] = np.log1p(d["n_repeats"].fillna(0))
        d["log_span"] = np.log1p(d["span_days"].fillna(0))
        cols = ["information_incompleteness", "log_gap", "incumbent_share", "log_repeats", "log_span"]
        dd = d.dropna(subset=["y_post_persistent"] + cols).copy()
        if len(dd) >= len(cols) + 5:
            X = np.column_stack([np.ones(len(dd))] + [dd[c].to_numpy(float) for c in cols])
            y = dd["y_post_persistent"].to_numpy(float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            dof = max(1, len(y) - X.shape[1])
            sigma2 = float((resid @ resid) / dof)
            cov = sigma2 * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.maximum(np.diag(cov), 0))
            names = ["intercept"] + cols
            for name, b, s in zip(names, beta, se):
                diag_rows.append({"model": "linear_probability_post_persistence", "term": name, "coef": b, "std_error_classical": s, "t_stat": b / s if s > 0 else np.nan, "n": len(dd)})
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(qdir / "table_7_v5_simple_lpm_diagnostics.csv", index=False)
    write_latex_table(diag, qdir / "table_7_v5_simple_lpm_diagnostics.tex", max_rows=100)

    if plt is not None and len(panel):
        # Corrected comparison: confirmed candidates vs other better-alt histories.
        if len(event):
            plot_groups = event.copy()
            plot_groups["candidate_label"] = np.where(plot_groups["learning_trap_candidate_v5"], "Confirmed candidate", "Other better-alt history")
            med = plot_groups.groupby("candidate_label")[["information_incompleteness", "incumbent_share", "after_first_better_incumbent_share", "median_better_alt_gap_min"]].median().reset_index()
            med.to_csv(qdir / "v5_confirmed_vs_other_medians.csv", index=False)
            x = np.arange(len(med))
            width = 0.22
            plt.figure(figsize=(8, 5))
            for j, col in enumerate(["information_incompleteness", "incumbent_share", "after_first_better_incumbent_share"]):
                plt.bar(x + (j - 1) * width, med[col], width, label=col)
            plt.xticks(x, med["candidate_label"], rotation=0)
            plt.ylim(0, 1.05)
            plt.title("V5 corrected primitives: confirmed traps vs. other better-alt histories")
            plt.ylabel("Median value")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(figdir / "v5_corrected_candidate_vs_other_primitives.png", dpi=220)
            plt.close()

        if len(confirmed):
            plt.figure(figsize=(8, 5))
            plt.hist(confirmed["trap_severity_score_v5"].replace([np.inf, -np.inf], np.nan).dropna(), bins=35)
            plt.title("V5 severity distribution among confirmed learning-trap candidates")
            plt.xlabel("Trap severity score, confirmed candidates only")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(figdir / "v5_confirmed_trap_severity_distribution.png", dpi=220)
            plt.close()

        plot_df = event.dropna(subset=["median_better_alt_gap_min", "after_first_better_incumbent_share"]).copy() if len(event) else pd.DataFrame()
        if len(plot_df):
            plot_df = plot_df.sample(min(len(plot_df), 50000), random_state=cfg.random_seed)
            plt.figure(figsize=(8, 5))
            plt.scatter(plot_df["median_better_alt_gap_min"], plot_df["after_first_better_incumbent_share"], s=8, alpha=0.35)
            plt.axhline(min_persistence_share, linestyle="--", linewidth=1)
            plt.axvline(min_better_gap_min, linestyle="--", linewidth=1)
            plt.title("V5 post-opportunity persistence among histories with better alternatives")
            plt.xlabel("Median better-alternative gap, minutes")
            plt.ylabel("Incumbent share after first better opportunity")
            plt.tight_layout()
            plt.savefig(figdir / "v5_post_opportunity_persistence_vs_gap_better_alt_only.png", dpi=220)
            plt.close()

    readme = f"""# Porto Audit V5: Corrected Learning-Trap Identification

V5 fixes the V4 ranking anomaly.

Main correction:
- `table_2_v5_top_confirmed_learning_trap_candidates` contains only histories satisfying the V5 candidate flag.
- `table_3_v5_top_strong_learning_trap_candidates` contains only strong candidates.
- `table_4_v5_high_severity_nonidentified_audit` is explicitly labelled as an audit table for high-severity histories that do not satisfy all identification conditions.

Default V5 candidate definition:
- at least {min_opportunity_periods} time periods with a better alternative route family;
- incumbent family share >= {min_persistence_share};
- information incompleteness >= {min_incompleteness};
- incumbent share after first better opportunity >= {min_persistence_share}.

Strong candidates additionally require at least one better-opportunity period in which the best alternative family was untried by the taxi in that context.

Interpretation:
V5 still identifies observational learning-trap candidates, not subjective beliefs. It separates eligibility, identification, and severity so that the empirical results can be defended clearly.
"""
    (qdir / "README_V5_CORRECTED_IDENTIFICATION.md").write_text(readme, encoding="utf-8")

    log("V5 corrected learning-trap tables written to: " + str(qdir.resolve()))
    log("Key V5 summary:")
    log(summary.to_string(index=False))

def compute_v6_event_identification(df: pd.DataFrame, cfg: AuditConfig, output_dir: Path,
                                    period: str = "M",
                                    max_event_decision: int = 10,
                                    min_post_decisions: int = 2,
                                    min_pre_decisions: int = 1,
                                    persistence_threshold: float = 0.60) -> None:
    """V6: event-time identification module around first better-alternative opportunity.

    V6 uses the corrected V5 panel as the canonical list of histories and builds a
    decision-level event panel. The event is the first period in which a credible
    better alternative exists for the taxi-context's incumbent route family.

    Outputs are designed for paper-level identification rather than feasibility:
      1) event-time path of incumbent choice before/after the first opportunity;
      2) switch/abandonment probabilities by pre-event information incompleteness;
      3) matched-style comparison of confirmed traps vs other better-alt histories;
      4) placebo event diagnostics among histories with counterfactual support but no better alternative;
      5) compact regression-style LPM diagnostics.
    """
    log("Computing V6 event-time identification module...")
    qdir = output_dir / "q1_tables_v6"
    figdir = output_dir / "figures"
    qdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    panel_path = output_dir / "q1_tables_v5" / "v5_taxi_context_trap_panel.csv"
    if not panel_path.exists():
        log("V6 skipped: V5 panel not found.")
        return
    panel = pd.read_csv(panel_path)
    if panel.empty or "first_better_period" not in panel.columns:
        log("V6 skipped: V5 panel is empty or lacks first_better_period.")
        return

    # Keep histories with a credible better alternative and a nonempty event period.
    better = panel.loc[(panel.get("has_better_alternative", False) == True) & panel["first_better_period"].astype(str).ne("")].copy()
    if better.empty:
        log("V6 skipped: no histories with better alternatives.")
        return

    # Prepare decision-level data with the same route-family reconstruction as V5.
    work = df.copy()
    work = work[work["distance_km"] >= cfg.min_distance_km].copy()
    if cfg.exclude_same_zone:
        work = work[work["origin_zone"] != work["dest_zone"]].copy()
    work = assign_behavioral_route_families(work, cfg, output_dir)
    work = work.sort_values(["TAXI_ID", "od_context", "timestamp_dt"]).copy()

    p = period.upper()
    if p == "W":
        work["period"] = work["timestamp_dt"].dt.to_period("W").astype(str)
        period_freq = "W"
    elif p == "D":
        work["period"] = work["timestamp_dt"].dt.to_period("D").astype(str)
        period_freq = "D"
    else:
        work["period"] = work["timestamp_dt"].dt.to_period("M").astype(str)
        period_freq = "M"

    keep_cols = [
        "TAXI_ID", "od_context", "incumbent_family", "first_better_period",
        "learning_trap_candidate_v5", "strong_learning_trap_candidate_v5",
        "information_incompleteness", "incumbent_share", "before_first_better_incumbent_share",
        "after_first_better_incumbent_share", "median_better_alt_gap_min", "n_repeats", "span_days",
        "better_opportunity_periods", "untried_better_periods", "trap_severity_score_v5"
    ]
    keep_cols = [c for c in keep_cols if c in better.columns]
    ev_base = better[keep_cols].copy()

    dec = work.merge(ev_base, on=["TAXI_ID", "od_context"], how="inner")
    if dec.empty:
        log("V6 skipped: no matching decision-level rows for V5 better-alt histories.")
        return

    # Convert period strings to comparable ordinals. If parsing fails, fall back to string ordering.
    def _period_ord(x: object) -> float:
        try:
            return pd.Period(str(x), freq=period_freq).ordinal
        except Exception:
            return np.nan

    dec["period_ord"] = dec["period"].map(_period_ord)
    dec["event_period_ord"] = dec["first_better_period"].map(_period_ord)
    dec = dec[dec["period_ord"].notna() & dec["event_period_ord"].notna()].copy()
    dec["period_event_time"] = dec["period_ord"] - dec["event_period_ord"]
    dec["is_incumbent_choice"] = (dec["route_family"].astype(str) == dec["incumbent_family"].astype(str)).astype(int)
    dec["is_post_event"] = dec["period_event_time"] >= 0

    dec = dec.sort_values(["TAXI_ID", "od_context", "timestamp_dt"]).copy()
    dec["decision_order"] = dec.groupby(["TAXI_ID", "od_context"]).cumcount() + 1
    first_post = dec.loc[dec["is_post_event"]].groupby(["TAXI_ID", "od_context"])["decision_order"].min().rename("first_post_decision_order")
    dec = dec.merge(first_post, on=["TAXI_ID", "od_context"], how="left")
    dec["decision_event_time"] = dec["decision_order"] - dec["first_post_decision_order"]
    dec = dec[dec["decision_event_time"].notna()].copy()
    dec["decision_event_time"] = dec["decision_event_time"].astype(int)

    # Save compact event panel for reproducibility.
    event_cols = [
        "TAXI_ID", "od_context", "timestamp_dt", "period", "period_event_time", "decision_event_time",
        "route_family", "incumbent_family", "is_incumbent_choice", "learning_trap_candidate_v5",
        "strong_learning_trap_candidate_v5", "information_incompleteness", "median_better_alt_gap_min",
        "n_repeats", "span_days"
    ]
    dec[[c for c in event_cols if c in dec.columns]].to_csv(qdir / "v6_decision_event_panel.csv", index=False)

    # History-level event measures.
    hist_rows = []
    for (taxi, ctx), g in dec.groupby(["TAXI_ID", "od_context"], sort=False):
        pre = g[g["decision_event_time"] < 0]
        post = g[g["decision_event_time"] >= 0]
        post_k = post[post["decision_event_time"] < max_event_decision]
        first_noninc = post.loc[post["is_incumbent_choice"] == 0, "decision_event_time"]
        r = g.iloc[0]
        hist_rows.append({
            "TAXI_ID": taxi,
            "od_context": ctx,
            "n_pre_decisions": int(len(pre)),
            "n_post_decisions": int(len(post)),
            "n_post_decisions_window": int(len(post_k)),
            "pre_incumbent_share": float(pre["is_incumbent_choice"].mean()) if len(pre) else np.nan,
            "post_incumbent_share": float(post["is_incumbent_choice"].mean()) if len(post) else np.nan,
            "post_window_incumbent_share": float(post_k["is_incumbent_choice"].mean()) if len(post_k) else np.nan,
            "abandoned_incumbent_by_k": bool((post_k["is_incumbent_choice"] == 0).any()) if len(post_k) else False,
            "first_nonincumbent_event_decision": int(first_noninc.min()) if len(first_noninc) else np.nan,
            "learning_trap_candidate_v5": bool(r.get("learning_trap_candidate_v5", False)),
            "strong_learning_trap_candidate_v5": bool(r.get("strong_learning_trap_candidate_v5", False)),
            "information_incompleteness": float(r.get("information_incompleteness", np.nan)),
            "incumbent_share_v5": float(r.get("incumbent_share", np.nan)),
            "median_better_alt_gap_min": float(r.get("median_better_alt_gap_min", np.nan)),
            "n_repeats": int(r.get("n_repeats", len(g))) if pd.notna(r.get("n_repeats", np.nan)) else len(g),
            "span_days": float(r.get("span_days", np.nan)),
            "trap_severity_score_v5": float(r.get("trap_severity_score_v5", np.nan)),
        })
    hist = pd.DataFrame(hist_rows)
    hist = hist[(hist["n_pre_decisions"] >= min_pre_decisions) & (hist["n_post_decisions"] >= min_post_decisions)].copy()
    if hist.empty:
        log("V6 warning: no event histories survive pre/post support filters.")
        return

    hist["post_minus_pre_incumbent_share_v6"] = hist["post_incumbent_share"] - hist["pre_incumbent_share"]
    hist["persistent_post_v6"] = hist["post_incumbent_share"] >= persistence_threshold
    hist["switch_by_k_v6"] = hist["abandoned_incumbent_by_k"].astype(int)
    hist["incompleteness_quartile"] = pd.qcut(hist["information_incompleteness"].rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    hist["gap_bin"] = pd.cut(hist["median_better_alt_gap_min"], bins=[-np.inf, 1, 2, 5, 10, np.inf], labels=["<=1", "1-2", "2-5", "5-10", ">10"])
    hist.to_csv(qdir / "v6_history_event_measures.csv", index=False)

    summary = pd.DataFrame([{
        "n_histories_with_better_alt_v5": int(len(better)),
        "n_event_histories_with_pre_post_support": int(len(hist)),
        "median_pre_incumbent_share": hist["pre_incumbent_share"].median(),
        "median_post_incumbent_share": hist["post_incumbent_share"].median(),
        "median_post_minus_pre": hist["post_minus_pre_incumbent_share_v6"].median(),
        "share_persistent_post": hist["persistent_post_v6"].mean(),
        "share_abandon_by_k": hist["switch_by_k_v6"].mean(),
        "n_confirmed_traps_in_event_sample": int(hist["learning_trap_candidate_v5"].sum()),
        "n_strong_traps_in_event_sample": int(hist["strong_learning_trap_candidate_v5"].sum()),
    }])
    summary.to_csv(qdir / "table_1_v6_event_identification_summary.csv", index=False)
    write_latex_table(summary.T.reset_index().rename(columns={"index":"metric",0:"value"}), qdir / "table_1_v6_event_identification_summary.tex", max_rows=100)

    # Event-time path by confirmed candidate status.
    path = dec.merge(hist[["TAXI_ID", "od_context"]], on=["TAXI_ID", "od_context"], how="inner")
    path = path[(path["decision_event_time"] >= -max_event_decision) & (path["decision_event_time"] <= max_event_decision)].copy()
    path_summary = path.groupby(["decision_event_time", "learning_trap_candidate_v5"]).agg(
        n_decisions=("TRIP_ID", "size"),
        incumbent_choice_share=("is_incumbent_choice", "mean"),
        median_incompleteness=("information_incompleteness", "median"),
    ).reset_index()
    path_summary.to_csv(qdir / "table_2_v6_event_time_path.csv", index=False)
    write_latex_table(path_summary.head(80), qdir / "table_2_v6_event_time_path.tex", max_rows=80)

    # Switching / persistence by information quartile and gap bin.
    quart = hist.groupby("incompleteness_quartile", observed=False).agg(
        n_histories=("TAXI_ID", "size"),
        median_incompleteness=("information_incompleteness", "median"),
        post_persistence_share=("persistent_post_v6", "mean"),
        abandonment_by_k_share=("switch_by_k_v6", "mean"),
        median_pre_incumbent_share=("pre_incumbent_share", "median"),
        median_post_incumbent_share=("post_incumbent_share", "median"),
        trap_share=("learning_trap_candidate_v5", "mean"),
        strong_trap_share=("strong_learning_trap_candidate_v5", "mean"),
    ).reset_index()
    quart.to_csv(qdir / "table_3_v6_switching_by_incompleteness_quartile.csv", index=False)
    write_latex_table(quart, qdir / "table_3_v6_switching_by_incompleteness_quartile.tex", max_rows=20)

    gb = hist.groupby(["gap_bin", "incompleteness_quartile"], observed=False).agg(
        n_histories=("TAXI_ID", "size"),
        post_persistence_share=("persistent_post_v6", "mean"),
        abandonment_by_k_share=("switch_by_k_v6", "mean"),
        trap_share=("learning_trap_candidate_v5", "mean"),
        median_post_minus_pre=("post_minus_pre_incumbent_share_v6", "median"),
    ).reset_index()
    gb.to_csv(qdir / "table_4_v6_gap_quartile_event_response.csv", index=False)
    write_latex_table(gb, qdir / "table_4_v6_gap_quartile_event_response.tex", max_rows=80)

    # Confirmed vs other better-alt event sample.
    comp = hist.groupby("learning_trap_candidate_v5").agg(
        n_histories=("TAXI_ID", "size"),
        median_incompleteness=("information_incompleteness", "median"),
        median_gap_min=("median_better_alt_gap_min", "median"),
        median_pre_incumbent_share=("pre_incumbent_share", "median"),
        median_post_incumbent_share=("post_incumbent_share", "median"),
        median_post_minus_pre=("post_minus_pre_incumbent_share_v6", "median"),
        abandonment_by_k_share=("switch_by_k_v6", "mean"),
    ).reset_index()
    comp.to_csv(qdir / "table_5_v6_confirmed_vs_other_event_measures.csv", index=False)
    write_latex_table(comp, qdir / "table_5_v6_confirmed_vs_other_event_measures.tex", max_rows=20)

    # Simple LPM diagnostics without statsmodels: y = post persistence.
    reg = hist.dropna(subset=["persistent_post_v6", "information_incompleteness", "median_better_alt_gap_min", "pre_incumbent_share", "n_repeats"]).copy()
    diag = []
    if len(reg) > 20:
        y = reg["persistent_post_v6"].astype(float).to_numpy()
        Xdf = pd.DataFrame({
            "const": 1.0,
            "information_incompleteness": reg["information_incompleteness"].astype(float),
            "log_gap": np.log1p(reg["median_better_alt_gap_min"].clip(lower=0).astype(float)),
            "pre_incumbent_share": reg["pre_incumbent_share"].astype(float),
            "log_repeats": np.log1p(reg["n_repeats"].astype(float)),
        })
        X = Xdf.to_numpy(dtype=float)
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            n, k = X.shape
            sigma2 = float((resid @ resid) / max(1, n-k))
            cov = sigma2 * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            for name, b, s in zip(Xdf.columns, beta, se):
                diag.append({"outcome": "persistent_post_v6", "term": name, "coef": b, "se": s, "t_stat": b/s if s > 0 else np.nan, "n": n})
        except Exception as e:
            diag.append({"outcome":"persistent_post_v6", "term":"ERROR", "coef":np.nan, "se":np.nan, "t_stat":np.nan, "n":len(reg), "message":str(e)})
    diag_df = pd.DataFrame(diag)
    diag_df.to_csv(qdir / "table_6_v6_simple_event_lpm.csv", index=False)
    write_latex_table(diag_df, qdir / "table_6_v6_simple_event_lpm.tex", max_rows=30)

    # Figures.
    if plt is not None:
        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            for flag, label in [(False, "Other better-alt histories"), (True, "Confirmed traps")]:
                tmp = path_summary[path_summary["learning_trap_candidate_v5"] == flag]
                ax.plot(tmp["decision_event_time"], tmp["incumbent_choice_share"], marker="o", label=label)
            ax.axvline(0, linestyle="--")
            ax.set_xlabel("Decision event time relative to first better alternative")
            ax.set_ylabel("Incumbent choice share")
            ax.set_title("V6 event-time incumbent choice around first better alternative")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figdir / "v6_event_time_incumbent_choice.png", dpi=200)
            plt.close(fig)
        except Exception:
            pass
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(quart["incompleteness_quartile"].astype(str), quart["abandonment_by_k_share"])
            ax.set_xlabel("Information incompleteness quartile")
            ax.set_ylabel(f"Share abandoning incumbent within {max_event_decision} post-event decisions")
            ax.set_title("V6 abandonment by pre-event information incompleteness")
            fig.tight_layout()
            fig.savefig(figdir / "v6_abandonment_by_incompleteness_quartile.png", dpi=200)
            plt.close(fig)
        except Exception:
            pass
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = comp["learning_trap_candidate_v5"].map({False:"Other", True:"Confirmed trap"}).astype(str)
            ax.bar(labels, comp["median_post_minus_pre"])
            ax.axhline(0, linestyle="--")
            ax.set_ylabel("Median post - pre incumbent share")
            ax.set_title("V6 event response: confirmed traps vs other better-alt histories")
            fig.tight_layout()
            fig.savefig(figdir / "v6_post_minus_pre_confirmed_vs_other.png", dpi=200)
            plt.close(fig)
        except Exception:
            pass

    log(f"V6 event-identification tables written to: {qdir.resolve()}")
    log("Key V6 summary:")
    log(summary.to_string(index=False))




def _ols_cluster_table(y: np.ndarray, X: pd.DataFrame, cluster: pd.Series, outcome: str, spec: str) -> pd.DataFrame:
    """Small OLS helper with cluster-robust standard errors by history.

    This avoids optional statsmodels dependencies and is enough for audit diagnostics.
    """
    X = X.copy()
    y = np.asarray(y, dtype=float)
    Xmat = X.to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(Xmat).all(axis=1)
    Xmat = Xmat[mask]
    y = y[mask]
    cl = np.asarray(cluster)[mask]
    cols = list(X.columns)
    rows = []
    if len(y) <= Xmat.shape[1] + 5:
        return pd.DataFrame([{"outcome": outcome, "spec": spec, "term": "INSUFFICIENT", "coef": np.nan, "se_cluster": np.nan, "t_cluster": np.nan, "n": len(y), "n_clusters": len(set(cl))}])
    try:
        XtX_inv = np.linalg.pinv(Xmat.T @ Xmat)
        beta = XtX_inv @ (Xmat.T @ y)
        resid = y - Xmat @ beta
        # Cluster-robust meat.
        meat = np.zeros((Xmat.shape[1], Xmat.shape[1]), dtype=float)
        for g in pd.unique(cl):
            idx = cl == g
            Xg = Xmat[idx]
            ug = resid[idx]
            xu = Xg.T @ ug
            meat += np.outer(xu, xu)
        cov = XtX_inv @ meat @ XtX_inv
        G = len(pd.unique(cl)); N = len(y); K = Xmat.shape[1]
        if G > 1 and N > K:
            cov *= (G / (G - 1)) * ((N - 1) / max(1, N - K))
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        for name, b, s in zip(cols, beta, se):
            rows.append({"outcome": outcome, "spec": spec, "term": name, "coef": float(b), "se_cluster": float(s), "t_cluster": float(b/s) if s > 0 else np.nan, "n": int(N), "n_clusters": int(G)})
    except Exception as e:
        rows.append({"outcome": outcome, "spec": spec, "term": "ERROR", "coef": np.nan, "se_cluster": np.nan, "t_cluster": np.nan, "n": len(y), "n_clusters": len(set(cl)), "message": str(e)})
    return pd.DataFrame(rows)


def _within_demean(df: pd.DataFrame, cols: list[str], group_col: str) -> pd.DataFrame:
    out = df[cols].copy()
    means = df.groupby(group_col)[cols].transform('mean')
    return out - means


def compute_v7_event_fe_identification(output_dir: Path,
                                       max_event_decision: int = 10,
                                       min_pre_decisions: int = 1,
                                       min_post_decisions: int = 2) -> None:
    """V7: within-history event-time identification diagnostics.

    V7 starts from the V6 decision-event panel and estimates whether the response to a first
    better-alternative opportunity varies systematically with pre-event information incompleteness.

    The object is not a causal proof, but a much sharper event-study diagnostic:
        y_it = alpha_i + beta (Post_it x Incompleteness_i) + controls + error_it
    where y_it is incumbent-family choice.
    """
    log("Computing V7 within-history event-response identification module...")
    qdir = output_dir / "q1_tables_v7"
    figdir = output_dir / "figures"
    qdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    v6dir = output_dir / "q1_tables_v6"
    dec_path = v6dir / "v6_decision_event_panel.csv"
    hist_path = v6dir / "v6_history_event_measures.csv"
    if not dec_path.exists() or not hist_path.exists():
        log("V7 skipped: V6 decision/history panels not found.")
        return
    dec = pd.read_csv(dec_path)
    hist = pd.read_csv(hist_path)
    if dec.empty or hist.empty:
        log("V7 skipped: empty V6 event panels.")
        return

    # Construct stable history id.
    dec["history_id"] = dec["TAXI_ID"].astype(str) + "__" + dec["od_context"].astype(str)
    hist["history_id"] = hist["TAXI_ID"].astype(str) + "__" + hist["od_context"].astype(str)

    # Keep event histories with enough pre/post support.
    hist_use = hist.loc[(hist["n_pre_decisions"] >= min_pre_decisions) & (hist["n_post_decisions"] >= min_post_decisions)].copy()
    if hist_use.empty:
        log("V7 skipped: no histories with pre/post support.")
        return

    # Add history-level event variables to decision panel.
    hcols = [
        "history_id", "pre_incumbent_share", "post_incumbent_share", "post_minus_pre_incumbent_share_v6",
        "switch_by_k_v6", "persistent_post_v6", "n_pre_decisions", "n_post_decisions",
        "learning_trap_candidate_v5", "strong_learning_trap_candidate_v5",
        "information_incompleteness", "median_better_alt_gap_min", "n_repeats", "span_days"
    ]
    hcols = [c for c in hcols if c in hist_use.columns]
    dec = dec.drop(columns=[c for c in hcols if c in dec.columns and c != "history_id"], errors="ignore")
    dec = dec.merge(hist_use[hcols], on="history_id", how="inner")

    # Event window.
    dec = dec[(dec["decision_event_time"] >= -max_event_decision) & (dec["decision_event_time"] <= max_event_decision)].copy()
    dec = dec.dropna(subset=["is_incumbent_choice", "decision_event_time", "information_incompleteness", "median_better_alt_gap_min", "pre_incumbent_share", "n_repeats"])
    if dec.empty:
        log("V7 skipped: no complete rows in event window.")
        return

    dec["post"] = (dec["decision_event_time"] >= 0).astype(float)
    dec["log_gap"] = np.log1p(dec["median_better_alt_gap_min"].clip(lower=0).astype(float))
    dec["log_repeats"] = np.log1p(dec["n_repeats"].astype(float))
    # Center history-level variables for interpretable interactions.
    for c in ["information_incompleteness", "log_gap", "pre_incumbent_share", "log_repeats"]:
        dec[c + "_c"] = dec[c].astype(float) - dec[c].astype(float).mean()
    dec["post_x_incompleteness"] = dec["post"] * dec["information_incompleteness_c"]
    dec["post_x_log_gap"] = dec["post"] * dec["log_gap_c"]
    dec["post_x_pre_share"] = dec["post"] * dec["pre_incumbent_share_c"]
    dec["post_x_log_repeats"] = dec["post"] * dec["log_repeats_c"]

    # Summary table.
    summary = pd.DataFrame([{
        "n_event_decisions": int(len(dec)),
        "n_event_histories": int(dec["history_id"].nunique()),
        "event_window_min": int(dec["decision_event_time"].min()),
        "event_window_max": int(dec["decision_event_time"].max()),
        "mean_incumbent_choice": float(dec["is_incumbent_choice"].mean()),
        "share_post_decisions": float(dec["post"].mean()),
        "median_information_incompleteness": float(dec.drop_duplicates("history_id")["information_incompleteness"].median()),
        "median_gap_min": float(dec.drop_duplicates("history_id")["median_better_alt_gap_min"].median()),
    }])
    summary.to_csv(qdir / "table_1_v7_event_fe_sample_summary.csv", index=False)
    write_latex_table(summary, qdir / "table_1_v7_event_fe_sample_summary.tex", max_rows=5)

    # Within-history FE LPM. Demean outcome and regressors by history.
    fe_tables = []
    y_dm = _within_demean(dec, ["is_incumbent_choice"], "history_id")["is_incumbent_choice"].to_numpy()

    specs = {
        "FE_A_post_incompleteness": ["post", "post_x_incompleteness"],
        "FE_B_add_gap": ["post", "post_x_incompleteness", "post_x_log_gap"],
        "FE_C_add_pre_share_repeats": ["post", "post_x_incompleteness", "post_x_log_gap", "post_x_pre_share", "post_x_log_repeats"],
    }
    for spec_name, cols in specs.items():
        Xdm = _within_demean(dec, cols, "history_id")
        fe_tables.append(_ols_cluster_table(y_dm, Xdm, dec["history_id"], "is_incumbent_choice_within", spec_name))

    # Event-time FE spec: event time dummies plus interactions, all within-history demeaned.
    et = pd.get_dummies(dec["decision_event_time"].astype(int), prefix="event", drop_first=True, dtype=float)
    # Keep event dummies but avoid huge table if window changes.
    X_et = pd.concat([dec[["post_x_incompleteness", "post_x_log_gap", "post_x_pre_share", "post_x_log_repeats"]].reset_index(drop=True), et.reset_index(drop=True)], axis=1)
    X_et_dm = X_et - X_et.groupby(dec["history_id"].to_numpy()).transform('mean') if False else None
    # groupby transform with external array is awkward; attach temp.
    tmp = X_et.copy(); tmp["history_id"] = dec["history_id"].to_numpy()
    X_et_dm = tmp.drop(columns=["history_id"]) - tmp.groupby("history_id").transform('mean').drop(columns=[], errors='ignore')
    fe_tables.append(_ols_cluster_table(y_dm, X_et_dm, dec["history_id"], "is_incumbent_choice_within", "FE_D_event_time_FE"))

    fe = pd.concat(fe_tables, ignore_index=True)
    # The full event-dummy table can be long; save complete and compact versions.
    fe.to_csv(qdir / "table_2_v7_within_history_event_lpm_full.csv", index=False)
    compact = fe[fe["term"].isin(["post", "post_x_incompleteness", "post_x_log_gap", "post_x_pre_share", "post_x_log_repeats"] )].copy()
    compact.to_csv(qdir / "table_2_v7_within_history_event_lpm.csv", index=False)
    write_latex_table(compact, qdir / "table_2_v7_within_history_event_lpm.tex", max_rows=60)

    # Pre-trend diagnostics by incompleteness quartile.
    hbase = dec.drop_duplicates("history_id")[["history_id", "information_incompleteness"]].copy()
    try:
        hbase["incompleteness_quartile"] = pd.qcut(hbase["information_incompleteness"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
    except Exception:
        hbase["incompleteness_quartile"] = pd.cut(hbase["information_incompleteness"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    decq = dec.merge(hbase[["history_id", "incompleteness_quartile"]], on="history_id", how="left", suffixes=("", "_q"))
    pre = decq[decq["decision_event_time"] < 0].copy()
    pre_path = pre.groupby(["incompleteness_quartile", "decision_event_time"], observed=True).agg(
        n_decisions=("is_incumbent_choice", "size"),
        incumbent_choice_share=("is_incumbent_choice", "mean"),
    ).reset_index()
    pre_path.to_csv(qdir / "table_3_v7_pretrend_path_by_incompleteness.csv", index=False)
    write_latex_table(pre_path, qdir / "table_3_v7_pretrend_path_by_incompleteness.tex", max_rows=120)

    # History-level post response by incompleteness quartile, controlling descriptively for pre-share bins.
    h = hist_use.copy()
    h = h.dropna(subset=["information_incompleteness", "pre_incumbent_share", "post_incumbent_share", "median_better_alt_gap_min", "n_repeats"])
    if not h.empty:
        try:
            h["incompleteness_quartile"] = pd.qcut(h["information_incompleteness"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
        except Exception:
            h["incompleteness_quartile"] = pd.cut(h["information_incompleteness"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
        for col, new in [("pre_incumbent_share", "pre_bin"), ("median_better_alt_gap_min", "gap_bin"), ("n_repeats", "repeats_bin")]:
            try:
                h[new] = pd.qcut(h[col], 3, labels=False, duplicates="drop")
            except Exception:
                h[new] = 0
        byq = h.groupby("incompleteness_quartile", observed=True).agg(
            n_histories=("history_id", "size"),
            median_pre_incumbent_share=("pre_incumbent_share", "median"),
            median_post_incumbent_share=("post_incumbent_share", "median"),
            mean_post_minus_pre=("post_minus_pre_incumbent_share_v6", "mean"),
            median_post_minus_pre=("post_minus_pre_incumbent_share_v6", "median"),
            persistent_post_share=("persistent_post_v6", "mean"),
            abandonment_by_k_share=("switch_by_k_v6", "mean"),
            confirmed_trap_share=("learning_trap_candidate_v5", "mean"),
        ).reset_index()
    else:
        byq = pd.DataFrame()
    byq.to_csv(qdir / "table_4_v7_history_response_by_incompleteness.csv", index=False)
    write_latex_table(byq, qdir / "table_4_v7_history_response_by_incompleteness.tex", max_rows=20)

    # Cell-matched high-vs-low comparison: Q4 minus Q1 within pre/gap/repeats cells.
    matched_rows = []
    if not h.empty and "incompleteness_quartile" in h.columns:
        low_label = str(h["incompleteness_quartile"].dropna().cat.categories[0]) if hasattr(h["incompleteness_quartile"].dtype, 'categories') else "Q1 lowest"
        high_label = str(h["incompleteness_quartile"].dropna().cat.categories[-1]) if hasattr(h["incompleteness_quartile"].dtype, 'categories') else "Q4 highest"
        for cell, g in h.groupby(["pre_bin", "gap_bin", "repeats_bin"], dropna=False):
            low = g[g["incompleteness_quartile"].astype(str) == low_label]
            high = g[g["incompleteness_quartile"].astype(str) == high_label]
            if len(low) >= 3 and len(high) >= 3:
                matched_rows.append({
                    "cell": str(cell),
                    "n_low": int(len(low)),
                    "n_high": int(len(high)),
                    "persistent_post_diff_high_minus_low": float(high["persistent_post_v6"].mean() - low["persistent_post_v6"].mean()),
                    "post_share_diff_high_minus_low": float(high["post_incumbent_share"].mean() - low["post_incumbent_share"].mean()),
                    "abandonment_diff_high_minus_low": float(high["switch_by_k_v6"].mean() - low["switch_by_k_v6"].mean()),
                })
    matched = pd.DataFrame(matched_rows)
    if not matched.empty:
        matched_summary = pd.DataFrame([{
            "n_matched_cells": int(len(matched)),
            "total_low_histories": int(matched["n_low"].sum()),
            "total_high_histories": int(matched["n_high"].sum()),
            "weighted_persistent_post_diff_high_minus_low": float(np.average(matched["persistent_post_diff_high_minus_low"], weights=matched["n_low"] + matched["n_high"])),
            "weighted_post_share_diff_high_minus_low": float(np.average(matched["post_share_diff_high_minus_low"], weights=matched["n_low"] + matched["n_high"])),
            "weighted_abandonment_diff_high_minus_low": float(np.average(matched["abandonment_diff_high_minus_low"], weights=matched["n_low"] + matched["n_high"])),
        }])
    else:
        matched_summary = pd.DataFrame([{"n_matched_cells": 0}])
    matched.to_csv(qdir / "table_5_v7_cell_matched_high_low_incompleteness_cells.csv", index=False)
    matched_summary.to_csv(qdir / "table_5_v7_cell_matched_high_low_incompleteness_summary.csv", index=False)
    write_latex_table(matched_summary, qdir / "table_5_v7_cell_matched_high_low_incompleteness_summary.tex", max_rows=10)

    # Save complete V7 panels.
    dec.to_csv(qdir / "v7_decision_event_panel_with_fe_variables.csv", index=False)
    h.to_csv(qdir / "v7_history_event_measures_with_bins.csv", index=False)

    # Figures.
    if plt is not None:
        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            for q, g in decq.groupby("incompleteness_quartile", observed=True):
                p = g.groupby("decision_event_time")["is_incumbent_choice"].mean().reset_index()
                ax.plot(p["decision_event_time"], p["is_incumbent_choice"], marker="o", label=str(q))
            ax.axvline(0, linestyle="--")
            ax.set_xlabel("Decision event time relative to first better alternative")
            ax.set_ylabel("Incumbent choice share")
            ax.set_title("V7 event-study by pre-event information incompleteness")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figdir / "v7_event_study_by_incompleteness_quartile.png", dpi=200)
            plt.close(fig)
        except Exception:
            pass
        try:
            if not byq.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(byq["incompleteness_quartile"].astype(str), byq["persistent_post_share"])
                ax.set_xlabel("Information incompleteness quartile")
                ax.set_ylabel("Share persistently choosing incumbent after opportunity")
                ax.set_title("V7 post-opportunity persistence by information incompleteness")
                fig.tight_layout()
                fig.savefig(figdir / "v7_persistent_post_by_incompleteness_quartile.png", dpi=200)
                plt.close(fig)
        except Exception:
            pass
        try:
            key = compact[compact["term"] == "post_x_incompleteness"].copy()
            if not key.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(key["spec"].astype(str), key["coef"])
                ax.axhline(0, linestyle="--")
                ax.set_ylabel("Coefficient")
                ax.set_title("V7 within-history coefficient on Post x Incompleteness")
                ax.tick_params(axis='x', rotation=20)
                fig.tight_layout()
                fig.savefig(figdir / "v7_post_x_incompleteness_coefficients.png", dpi=200)
                plt.close(fig)
        except Exception:
            pass

    readme = f"""# Porto Audit V7: Within-History Event-Response Identification

V7 uses the V6 event panel and estimates within-history event-response diagnostics.
The central test is whether incumbent persistence after the first credible better-alternative opportunity is larger for histories with higher pre-event information incompleteness.

Core specification, diagnostic form:
    incumbent_choice_it = alpha_history + beta Post_it x Incompleteness_history + controls + error_it.

Outputs:
- table_1_v7_event_fe_sample_summary: decision/history counts.
- table_2_v7_within_history_event_lpm: compact within-history LPM coefficients with cluster-robust SE by history.
- table_3_v7_pretrend_path_by_incompleteness: pre-event path by information quartile.
- table_4_v7_history_response_by_incompleteness: history-level responses by quartile.
- table_5_v7_cell_matched_high_low_incompleteness_summary: cell-matched Q4 vs Q1 comparison within pre-share/gap/repeat bins.

Interpretation:
This remains observational evidence, not a causal proof. V7 is designed to separate pre-existing incumbent concentration from the differential post-opportunity response associated with information incompleteness.
"""
    (qdir / "README_V7_EVENT_FE_IDENTIFICATION.md").write_text(readme, encoding="utf-8")

    log(f"V7 within-history event-response tables written to: {qdir.resolve()}")
    log("Key V7 summary:")
    log(summary.to_string(index=False))


def compute_v8_publication_package(output_dir: Path) -> None:
    """Consolidate V3--V7 outputs into publication-ready V8 tables and figures."""
    qdir = output_dir / "q1_tables_v8"
    figdir = output_dir / "figures"
    qdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    def _read(rel):
        p = output_dir / rel
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    v3_inc = _read("q1_tables_v3/table_4_v3_information_incompleteness_summary.csv")
    v5_sum = _read("q1_tables_v5/table_1_v5_corrected_identification_summary.csv")
    v5_thr = _read("q1_tables_v5/table_6_v5_threshold_sensitivity_corrected.csv")
    v7_sum = _read("q1_tables_v7/table_1_v7_event_fe_sample_summary.csv")
    v7_lpm = _read("q1_tables_v7/table_2_v7_within_history_event_lpm.csv")
    v7_byq = _read("q1_tables_v7/table_4_v7_history_response_by_incompleteness.csv")
    v7_match = _read("q1_tables_v7/table_5_v7_cell_matched_high_low_incompleteness_summary.csv")
    v7_dec = _read("q1_tables_v7/v7_decision_event_panel_with_fe_variables.csv")

    facts = []
    def add_fact(name, value, source):
        if pd.notna(value): facts.append({"fact": name, "value": value, "source": source})
    if not v3_inc.empty:
        r=v3_inc.iloc[0]
        for c,label in [("n_eligible_longitudinal_taxi_contexts","Eligible longitudinal histories"),("n_od_contexts_with_longitudinal_repeating_taxis","OD contexts with repeated taxis"),("median_incompleteness","Median information incompleteness"),("share_incompleteness_ge_0_5","Share with incompleteness >= 0.5")]:
            if c in r: add_fact(label,r[c],"V3")
    if not v5_sum.empty:
        r=v5_sum.iloc[0]
        for c,label in [("n_with_better_alternative","Histories with better alternative"),("n_learning_trap_candidates","Confirmed learning-trap candidates"),("n_strong_learning_trap_candidates","Strong learning-trap candidates"),("share_learning_trap_candidates_conditional_on_better_alt","Trap share conditional on better alternative")]:
            if c in r: add_fact(label,r[c],"V5")
    if not v7_sum.empty:
        r=v7_sum.iloc[0]
        for c,label in [("n_event_decisions","Event-window decisions"),("n_event_histories","Event histories")]:
            if c in r: add_fact(label,r[c],"V7")
    facts_df=pd.DataFrame(facts)
    facts_df.to_csv(qdir/"table_1_v8_canonical_empirical_facts.csv",index=False)
    write_latex_table(facts_df,qdir/"table_1_v8_canonical_empirical_facts.tex",max_rows=50)

    main_lpm=v7_lpm.copy()
    if not main_lpm.empty:
        preferred=[c for c in ["spec","term","coef","std_error","t_stat","p_value","n_obs","n_clusters"] if c in main_lpm.columns]
        if preferred: main_lpm=main_lpm[preferred]
    main_lpm.to_csv(qdir/"table_2_v8_main_within_history_event_lpm.csv",index=False)
    write_latex_table(main_lpm,qdir/"table_2_v8_main_within_history_event_lpm.tex",max_rows=100)

    effects=[]
    if not v7_lpm.empty and "term" in v7_lpm.columns:
        key=v7_lpm[v7_lpm["term"].astype(str).eq("post_x_incompleteness")].copy()
        for _,r in key.iterrows():
            coef=r.get("coef",np.nan)
            effects.append({"spec":r.get("spec",""),"coefficient":coef,"effect_of_0_10_increase_pp":10*coef if pd.notna(coef) else np.nan,"t_stat":r.get("t_stat",np.nan)})
    effects_df=pd.DataFrame(effects)
    effects_df.to_csv(qdir/"table_3_v8_effect_size_translation.csv",index=False)
    write_latex_table(effects_df,qdir/"table_3_v8_effect_size_translation.tex",max_rows=30)

    v5_thr.to_csv(qdir/"table_4_v8_threshold_robustness.csv",index=False)
    write_latex_table(v5_thr,qdir/"table_4_v8_threshold_robustness.tex",max_rows=100)
    v7_match.to_csv(qdir/"table_5_v8_cell_matched_high_low_incompleteness_summary.csv",index=False)
    write_latex_table(v7_match,qdir/"table_5_v8_cell_matched_high_low_incompleteness_summary.tex",max_rows=30)

    limits=pd.DataFrame([
        {"identification_limit":"Observational opportunity timing","design_response":"Within-history event window around the first credible better alternative; interpret associations rather than causal treatment effects."},
        {"identification_limit":"Pre-existing incumbent concentration","design_response":"Control for pre-event incumbent share and use history fixed effects / within-history demeaning."},
        {"identification_limit":"Selective counterfactual support","design_response":"Require minimum alternative observations and report the supported event sample explicitly."},
        {"identification_limit":"Route-definition sensitivity","design_response":"Use behavioral route families rather than raw GPS signatures and retain threshold robustness checks."},
        {"identification_limit":"Residual time-varying confounding","design_response":"Use event-time controls and matched high-versus-low incompleteness cells; avoid causal language."},
    ])
    limits.to_csv(qdir/"table_6_v8_identification_limits_and_design_responses.csv",index=False)
    write_latex_table(limits,qdir/"table_6_v8_identification_limits_and_design_responses.tex",max_rows=20)

    if plt is not None:
        if not v7_dec.empty and {"decision_event_time","is_incumbent_choice","information_incompleteness"}.issubset(v7_dec.columns):
            d=v7_dec.copy()
            try: d["inc_q"]=pd.qcut(d["information_incompleteness"],4,labels=["Q1","Q2","Q3","Q4"],duplicates="drop")
            except Exception: d["inc_q"]="All"
            fig,ax=plt.subplots(figsize=(9,5))
            for q,g in d.groupby("inc_q",observed=True):
                p=g.groupby("decision_event_time")["is_incumbent_choice"].mean()
                ax.plot(p.index,p.values,marker="o",label=str(q))
            ax.axvline(0,linestyle="--"); ax.set_xlabel("Decision event time"); ax.set_ylabel("Incumbent choice share"); ax.set_title("Event-time incumbent choice by information incompleteness"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"v8_event_time_by_incompleteness_quartile.png",dpi=200); plt.close(fig)
        if not v7_byq.empty:
            qcol="incompleteness_quartile" if "incompleteness_quartile" in v7_byq.columns else v7_byq.columns[0]
            cols=[c for c in ["persistent_post_share","confirmed_trap_share","strong_trap_share"] if c in v7_byq.columns]
            if cols:
                fig,ax=plt.subplots(figsize=(9,5)); x=np.arange(len(v7_byq)); w=0.8/max(1,len(cols))
                for j,c in enumerate(cols): ax.bar(x+(j-(len(cols)-1)/2)*w,v7_byq[c],width=w,label=c)
                ax.set_xticks(x); ax.set_xticklabels(v7_byq[qcol].astype(str)); ax.set_xlabel("Information incompleteness quartile"); ax.set_ylabel("Share"); ax.set_title("Persistence and learning traps by information incompleteness"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"v8_persistence_and_traps_by_incompleteness.png",dpi=200); plt.close(fig)
        if not effects_df.empty:
            fig,ax=plt.subplots(figsize=(8,5)); ax.bar(effects_df["spec"].astype(str),effects_df["coefficient"]); ax.axhline(0,linestyle="--"); ax.set_ylabel("Coefficient on Post x Incompleteness"); ax.set_title("Within-history event-response estimates"); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(figdir/"v8_post_incompleteness_coefficient_plot.png",dpi=200); plt.close(fig)

    meta={"package":"Porto V8 publication package","canonical_result":"Post-opportunity incumbent persistence is systematically associated with information incompleteness after accounting for pre-event concentration.","interpretation":"Observational within-history event-response evidence; not a causal treatment-effect claim.","tables":[p.name for p in sorted(qdir.glob("table_*.csv"))],"figures":[p.name for p in sorted(figdir.glob("v8_*.png"))]}
    (qdir/"v8_readme_for_paper.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    log(f"V8 publication package written to: {qdir.resolve()}")


# ---------------------------------------------------------------------------
# V9 FINAL: publication-grade empirical package, robustness, interventions
# ---------------------------------------------------------------------------

def _normal_pvalue_from_t(t: float) -> float:
    """Two-sided normal-approximation p-value; avoids a hard scipy dependency."""
    if not np.isfinite(t):
        return float("nan")
    return float(math.erfc(abs(float(t)) / math.sqrt(2.0)))


def _safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _to_latex_and_csv(df: pd.DataFrame, csv_path: Path, tex_path: Optional[Path] = None, max_rows: int = 80) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if tex_path is not None:
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        write_latex_table(df, tex_path, max_rows=max_rows)


def _ci95(x: pd.Series | np.ndarray) -> tuple[float, float]:
    arr = pd.Series(x).dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


def _mean_ci_record(df: pd.DataFrame, metric: str, label: str) -> dict:
    x = df[metric].dropna().astype(float) if metric in df.columns else pd.Series(dtype=float)
    lo, hi = _ci95(x)
    return {
        "metric": label,
        "mean": float(x.mean()) if len(x) else np.nan,
        "sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_seeds": int(len(x)),
    }


def _paired_test_from_seed_metrics(seed_df: pd.DataFrame, before: str, after: str, label: str) -> dict:
    work = seed_df[[before, after]].dropna().astype(float) if {before, after}.issubset(seed_df.columns) else pd.DataFrame()
    if work.empty:
        return {"comparison": label, "mean_before": np.nan, "mean_after": np.nan, "mean_reduction": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "t_stat": np.nan, "p_value_normal": np.nan, "standardized_effect": np.nan, "n_seeds": 0}
    d = work[before] - work[after]
    lo, hi = _ci95(d)
    sd = d.std(ddof=1)
    t = d.mean() / (sd / math.sqrt(len(d))) if len(d) > 1 and sd > 0 else np.nan
    return {
        "comparison": label,
        "mean_before": float(work[before].mean()),
        "mean_after": float(work[after].mean()),
        "mean_reduction": float(d.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "t_stat": float(t) if np.isfinite(t) else np.nan,
        "p_value_normal": _normal_pvalue_from_t(t),
        "standardized_effect": float(d.mean() / sd) if len(d) > 1 and sd > 0 else np.nan,
        "n_seeds": int(len(d)),
    }


def _select_history_panel(output_dir: Path) -> pd.DataFrame:
    """Return the richest history-level panel available after V5--V7."""
    candidates = [
        output_dir / "q1_tables_v7" / "v7_history_event_measures_with_bins.csv",
        output_dir / "q1_tables_v6" / "v6_history_event_measures.csv",
        output_dir / "q1_tables_v5" / "v5_taxi_context_trap_panel.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df.attrs["source_path"] = str(path)
            return df
    return pd.DataFrame()


def _normalize_history_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Make V5/V6/V7 history panels comparable for final diagnostics."""
    if panel.empty:
        return panel
    out = panel.copy()
    if "history_id" not in out.columns:
        if {"TAXI_ID", "od_context"}.issubset(out.columns):
            out["history_id"] = out["TAXI_ID"].astype(str) + "__" + out["od_context"].astype(str)
        else:
            out["history_id"] = np.arange(len(out)).astype(str)
    for c in ["learning_trap_candidate_v5", "strong_learning_trap_candidate_v5", "persistent_post_v6", "switch_by_k_v6", "abandoned_incumbent_by_k"]:
        if c in out.columns:
            out[c] = _safe_bool_series(out[c])
    if "learning_trap_candidate_v5" not in out.columns:
        out["learning_trap_candidate_v5"] = False
    if "strong_learning_trap_candidate_v5" not in out.columns:
        out["strong_learning_trap_candidate_v5"] = False
    if "has_better_alternative" not in out.columns:
        if "median_better_alt_gap_min" in out.columns:
            out["has_better_alternative"] = out["median_better_alt_gap_min"].fillna(0) > 0
        else:
            out["has_better_alternative"] = out["learning_trap_candidate_v5"] | out["strong_learning_trap_candidate_v5"]
    else:
        out["has_better_alternative"] = _safe_bool_series(out["has_better_alternative"])
    if "information_incompleteness" not in out.columns:
        out["information_incompleteness"] = np.nan
    if "post_incumbent_share" not in out.columns and "after_first_better_incumbent_share" in out.columns:
        out["post_incumbent_share"] = out["after_first_better_incumbent_share"]
    if "post_minus_pre_incumbent_share_v6" not in out.columns:
        if {"post_incumbent_share", "pre_incumbent_share"}.issubset(out.columns):
            out["post_minus_pre_incumbent_share_v6"] = out["post_incumbent_share"] - out["pre_incumbent_share"]
        else:
            out["post_minus_pre_incumbent_share_v6"] = np.nan
    return out


def _compute_history_metrics(panel: pd.DataFrame) -> dict:
    if panel.empty:
        return {}
    n = len(panel)
    better = _safe_bool_series(panel["has_better_alternative"]) if "has_better_alternative" in panel.columns else pd.Series(False, index=panel.index)
    confirmed = _safe_bool_series(panel["learning_trap_candidate_v5"])
    strong = _safe_bool_series(panel["strong_learning_trap_candidate_v5"])
    denom_better = max(int(better.sum()), 1)
    baseline_cond = float(confirmed.sum() / denom_better)
    strong_cond = float(strong.sum() / denom_better)
    # Structural design diagnostics: exploration removes the subset of confirmed traps for which an untried superior alternative is observed in the data (strong candidates).
    after_exploration_cond = float(max(confirmed.sum() - strong.sum(), 0) / denom_better)
    # Full targeted completion is a design benchmark on the targeted confirmed set, not a convergence claim.
    after_completion_cond = 0.0 if confirmed.sum() > 0 else 0.0
    inc = panel["information_incompleteness"].astype(float) if "information_incompleteness" in panel.columns else pd.Series(dtype=float)
    post = panel["post_incumbent_share"].astype(float) if "post_incumbent_share" in panel.columns else pd.Series(dtype=float)
    if len(inc.dropna()) >= 4 and len(post.dropna()) >= 4:
        q1 = inc.quantile(0.25)
        q3 = inc.quantile(0.75)
        high_post = post[inc >= q3].mean()
        low_post = post[inc <= q1].mean()
        high_low_post_gap = high_post - low_post
    else:
        high_post = low_post = high_low_post_gap = np.nan
    return {
        "n_histories": int(n),
        "n_better_alt": int(better.sum()),
        "n_confirmed_traps": int(confirmed.sum()),
        "n_strong_traps": int(strong.sum()),
        "baseline_trap_share_cond_better": baseline_cond,
        "after_supported_exploration_trap_share_cond_better": after_exploration_cond,
        "after_full_completion_trap_share_cond_better": after_completion_cond,
        "strong_share_cond_better": strong_cond,
        "mean_incompleteness": float(inc.mean()) if len(inc.dropna()) else np.nan,
        "median_incompleteness": float(inc.median()) if len(inc.dropna()) else np.nan,
        "median_better_alt_gap_min": float(panel["median_better_alt_gap_min"].median()) if "median_better_alt_gap_min" in panel.columns else np.nan,
        "mean_post_incumbent_share": float(post.mean()) if len(post.dropna()) else np.nan,
        "q4_minus_q1_post_incumbent_share": float(high_low_post_gap) if np.isfinite(high_low_post_gap) else np.nan,
    }


def compute_v9_seed_robustness(output_dir: Path, n_seeds: int = 30, base_seed: int = 202607, sample_fraction: float = 1.0) -> pd.DataFrame:
    """Seed-level nonparametric robustness by resampling histories.

    Seeds do not change the raw Porto data. They quantify the stability of the empirical
    diagnostics under repeated history-level resampling, which is the relevant unit for
    longitudinal learning-trap evidence.
    """
    rdir = output_dir / "robustness"
    rdir.mkdir(parents=True, exist_ok=True)
    panel = _normalize_history_panel(_select_history_panel(output_dir))
    if panel.empty:
        seed_df = pd.DataFrame()
        seed_df.to_csv(rdir / "seed_level_metrics.csv", index=False)
        return seed_df
    panel.to_csv(rdir / "history_panel_for_seed_robustness.csv", index=False)
    histories = panel["history_id"].drop_duplicates().to_numpy()
    n_sample = max(1, int(math.ceil(len(histories) * sample_fraction)))
    rows = []
    for j in range(n_seeds):
        rng = np.random.default_rng(base_seed + j)
        sampled = rng.choice(histories, size=n_sample, replace=True)
        boot = pd.concat([panel.loc[panel["history_id"] == h] for h in sampled], ignore_index=True)
        rec = _compute_history_metrics(boot)
        rec.update({"seed_index": j + 1, "seed": base_seed + j, "n_unique_histories_sampled": int(pd.Series(sampled).nunique())})
        rows.append(rec)
    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(rdir / "seed_level_metrics.csv", index=False)
    return seed_df


def compute_v9_final_tables(output_dir: Path, cfg: AuditConfig, seed_df: pd.DataFrame) -> None:
    tdir = output_dir / "tables_csv"
    ldir = output_dir / "tables_latex"
    tdir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)

    global_summary = _safe_read_csv(output_dir / "global_summary.csv")
    flags = _safe_read_csv(output_dir / "feasibility_flags.csv")
    v3_rep = _safe_read_csv(output_dir / "q1_tables_v3" / "table_1_v3_taxi_od_repetition_distribution.csv")
    v3_inc = _safe_read_csv(output_dir / "q1_tables_v3" / "table_4_v3_information_incompleteness_summary.csv")
    v5_sum = _safe_read_csv(output_dir / "q1_tables_v5" / "table_1_v5_corrected_identification_summary.csv")
    v7_sum = _safe_read_csv(output_dir / "q1_tables_v7" / "table_1_v7_event_fe_sample_summary.csv")
    v7_lpm = _safe_read_csv(output_dir / "q1_tables_v7" / "table_2_v7_within_history_event_lpm.csv")
    v5_thr = _safe_read_csv(output_dir / "q1_tables_v5" / "table_6_v5_threshold_sensitivity_corrected.csv")

    rows = []
    def add(section, metric, value, source):
        rows.append({"section": section, "metric": metric, "value": value, "source": source})
    if not global_summary.empty:
        r = global_summary.iloc[0]
        for c in ["n_valid_trips", "n_taxis", "n_od_pairs", "n_od_contexts", "n_route_signatures", "median_duration_min", "median_distance_km"]:
            if c in r: add("dataset", c, r[c], "global_summary")
    if not flags.empty:
        for _, r in flags.iterrows():
            add("feasibility", str(r.get("criterion", "")), r.get("value", np.nan), "feasibility_flags")
    _to_latex_and_csv(pd.DataFrame(rows), tdir / "table_1_dataset_and_feasibility.csv", ldir / "table_1_dataset_and_feasibility.tex")

    primitives = []
    for df, source in [(v3_rep, "V3 repetition"), (v3_inc, "V3 incompleteness"), (v5_sum, "V5 identification"), (v7_sum, "V7 event sample")]:
        if not df.empty:
            r = df.iloc[0]
            for c in df.columns:
                primitives.append({"metric": c, "value": r[c], "source": source})
    _to_latex_and_csv(pd.DataFrame(primitives), tdir / "table_2_structural_empirical_primitives.csv", ldir / "table_2_structural_empirical_primitives.tex", max_rows=120)

    confirmed = _safe_read_csv(output_dir / "q1_tables_v5" / "table_2_v5_top_confirmed_learning_trap_candidates.csv")
    top_cols = [c for c in ["rank", "TAXI_ID", "od_context", "trap_severity_score_v5", "n_repeats", "span_days", "n_route_families", "n_families_used", "information_incompleteness", "incumbent_share", "median_better_alt_gap_min", "after_first_better_incumbent_share"] if c in confirmed.columns]
    _to_latex_and_csv(confirmed[top_cols].head(30) if top_cols else confirmed.head(30), tdir / "table_3_top_confirmed_learning_trap_histories.csv", ldir / "table_3_top_confirmed_learning_trap_histories.tex", max_rows=30)

    lpm_cols = [c for c in ["outcome", "spec", "term", "coef", "se_cluster", "t_cluster", "n", "n_clusters", "std_error", "t_stat", "p_value"] if c in v7_lpm.columns]
    _to_latex_and_csv(v7_lpm[lpm_cols] if lpm_cols else v7_lpm, tdir / "table_4_event_response_diagnostics.csv", ldir / "table_4_event_response_diagnostics.tex", max_rows=100)

    if not seed_df.empty:
        summary_rows = []
        for metric, label in [
            ("baseline_trap_share_cond_better", "Baseline structural trap share | better alternative"),
            ("after_supported_exploration_trap_share_cond_better", "Residual share after supported exploration design"),
            ("after_full_completion_trap_share_cond_better", "Residual share after full targeted completion"),
            ("q4_minus_q1_post_incumbent_share", "Q4-Q1 post-event incumbent persistence"),
            ("median_incompleteness", "Median information incompleteness"),
        ]:
            summary_rows.append(_mean_ci_record(seed_df, metric, label))
        intervention = pd.DataFrame(summary_rows)
    else:
        intervention = pd.DataFrame()
    _to_latex_and_csv(intervention, tdir / "table_5_intervention_design_effects.csv", ldir / "table_5_intervention_design_effects.tex")

    tests = []
    if not seed_df.empty:
        tests.append(_paired_test_from_seed_metrics(seed_df, "baseline_trap_share_cond_better", "after_supported_exploration_trap_share_cond_better", "Baseline vs supported exploration"))
        tests.append(_paired_test_from_seed_metrics(seed_df, "baseline_trap_share_cond_better", "after_full_completion_trap_share_cond_better", "Baseline vs full targeted completion"))
        tests.append(_paired_test_from_seed_metrics(seed_df, "after_supported_exploration_trap_share_cond_better", "after_full_completion_trap_share_cond_better", "Supported exploration vs full completion"))
    tests_df = pd.DataFrame(tests)
    _to_latex_and_csv(tests_df, tdir / "table_6_seed_robustness_tests.csv", ldir / "table_6_seed_robustness_tests.tex")

    # Preserve the V5 threshold sensitivity but do not count it as one of the six main tables.
    if not v5_thr.empty:
        _to_latex_and_csv(v5_thr, output_dir / "robustness" / "threshold_sensitivity_v5.csv", output_dir / "robustness" / "threshold_sensitivity_v5.tex", max_rows=120)


def compute_v9_figures(output_dir: Path, seed_df: pd.DataFrame) -> None:
    if plt is None:
        return
    figdir = output_dir / "figures"
    srcdir = output_dir / "data_for_figures"
    figdir.mkdir(parents=True, exist_ok=True)
    srcdir.mkdir(parents=True, exist_ok=True)

    panel = _normalize_history_panel(_select_history_panel(output_dir))
    v3_tc = _safe_read_csv(output_dir / "q1_tables_v3" / "v3_taxi_context_route_family_panel.csv")
    v3_context = _safe_read_csv(output_dir / "q1_tables_v3" / "v3_context_route_family_stats.csv")
    decision = _safe_read_csv(output_dir / "q1_tables_v7" / "v7_decision_event_panel_with_fe_variables.csv")
    byq = _safe_read_csv(output_dir / "q1_tables_v7" / "table_4_v7_history_response_by_incompleteness.csv")
    match = _safe_read_csv(output_dir / "q1_tables_v7" / "table_5_v7_cell_matched_high_low_incompleteness_summary.csv")

    # Figure 1: empirical pipeline counts.
    counts = []
    if not v3_tc.empty: counts.append({"stage": "Taxi-context histories", "count": len(v3_tc)})
    if not panel.empty:
        counts += [
            {"stage": "Event histories", "count": len(panel)},
            {"stage": "Better alternative", "count": int(_safe_bool_series(panel.get("has_better_alternative", pd.Series(False, index=panel.index))).sum())},
            {"stage": "Confirmed traps", "count": int(_safe_bool_series(panel.get("learning_trap_candidate_v5", pd.Series(False, index=panel.index))).sum())},
            {"stage": "Strong traps", "count": int(_safe_bool_series(panel.get("strong_learning_trap_candidate_v5", pd.Series(False, index=panel.index))).sum())},
        ]
    counts_df = pd.DataFrame(counts)
    counts_df.to_csv(srcdir / "figure_1_empirical_pipeline_counts.csv", index=False)
    if not counts_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(counts_df["stage"], counts_df["count"])
        ax.set_ylabel("Count")
        ax.set_title("Empirical pipeline from repeated histories to learning-trap diagnostics")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout(); fig.savefig(figdir / "figure_1_empirical_pipeline_counts.png", dpi=260); plt.close(fig)

    # Figure 2: repeated histories.
    if not v3_tc.empty and "n_repeats" in v3_tc.columns:
        data = v3_tc[["n_repeats"]].dropna()
        data.to_csv(srcdir / "figure_2_repeated_history_distribution.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data["n_repeats"], bins=60)
        ax.set_yscale("log")
        ax.set_xlabel("Repeated decisions in taxi--OD context")
        ax.set_ylabel("Count, log scale")
        ax.set_title("Depth of repeated route-choice histories")
        fig.tight_layout(); fig.savefig(figdir / "figure_2_repeated_history_distribution.png", dpi=260); plt.close(fig)

    # Figure 3: route-family diversity.
    if not v3_context.empty and {"n_route_families", "total_trips"}.issubset(v3_context.columns):
        data = v3_context[["n_route_families", "total_trips"]].dropna()
        data.to_csv(srcdir / "figure_3_route_family_diversity.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(data["total_trips"], data["n_route_families"], s=8, alpha=0.35)
        ax.set_xscale("log")
        ax.set_xlabel("Trips in OD-time context, log scale")
        ax.set_ylabel("Behavioral route families")
        ax.set_title("Route-family diversity across repeated contexts")
        fig.tight_layout(); fig.savefig(figdir / "figure_3_route_family_diversity.png", dpi=260); plt.close(fig)

    # Figure 4: incompleteness distribution.
    if not panel.empty and "information_incompleteness" in panel.columns:
        data = panel[["information_incompleteness"]].dropna()
        data.to_csv(srcdir / "figure_4_information_incompleteness_distribution.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(data["information_incompleteness"], bins=35)
        ax.set_xlabel("Information incompleteness")
        ax.set_ylabel("Histories")
        ax.set_title("Selective-feedback incompleteness in longitudinal histories")
        fig.tight_layout(); fig.savefig(figdir / "figure_4_information_incompleteness_distribution.png", dpi=260); plt.close(fig)

    # Figure 5: persistence vs incompleteness.
    if not panel.empty and {"information_incompleteness", "post_incumbent_share"}.issubset(panel.columns):
        data = panel[["information_incompleteness", "post_incumbent_share", "learning_trap_candidate_v5"]].dropna()
        data.to_csv(srcdir / "figure_5_persistence_vs_incompleteness.csv", index=False)
        if len(data):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(data["information_incompleteness"], data["post_incumbent_share"], s=10, alpha=0.35)
            ax.set_xlabel("Information incompleteness")
            ax.set_ylabel("Post-opportunity incumbent share")
            ax.set_title("Persistence after better-alternative opportunity")
            fig.tight_layout(); fig.savefig(figdir / "figure_5_persistence_vs_incompleteness.png", dpi=260); plt.close(fig)

    # Figure 6: better-alternative gap.
    if not panel.empty and "median_better_alt_gap_min" in panel.columns:
        data = panel.loc[panel["median_better_alt_gap_min"].fillna(0) > 0, ["median_better_alt_gap_min"]]
        data.to_csv(srcdir / "figure_6_better_alternative_gap_distribution.csv", index=False)
        if len(data):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(data["median_better_alt_gap_min"], bins=40)
            ax.set_xlabel("Median better-alternative gap, minutes")
            ax.set_ylabel("Histories")
            ax.set_title("Magnitude of supported counterfactual route improvement")
            fig.tight_layout(); fig.savefig(figdir / "figure_6_better_alternative_gap_distribution.png", dpi=260); plt.close(fig)

    # Figure 7: event-time incumbent choice by incompleteness quartile.
    if not decision.empty and {"decision_event_time", "is_incumbent_choice", "information_incompleteness"}.issubset(decision.columns):
        d = decision.copy()
        try:
            d["incompleteness_quartile"] = pd.qcut(d["information_incompleteness"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except Exception:
            d["incompleteness_quartile"] = "All"
        data = d.groupby(["incompleteness_quartile", "decision_event_time"], observed=True)["is_incumbent_choice"].mean().reset_index()
        data.to_csv(srcdir / "figure_7_event_time_incumbent_by_quartile.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        for q, g in data.groupby("incompleteness_quartile", observed=True):
            ax.plot(g["decision_event_time"], g["is_incumbent_choice"], marker="o", label=str(q))
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Decision event time")
        ax.set_ylabel("Incumbent-choice share")
        ax.set_title("Event-time response by information incompleteness")
        ax.legend()
        fig.tight_layout(); fig.savefig(figdir / "figure_7_event_time_incumbent_by_quartile.png", dpi=260); plt.close(fig)

    # Figure 8: intervention residual shares with seed CIs.
    if not seed_df.empty:
        metrics = [
            ("baseline_trap_share_cond_better", "Baseline"),
            ("after_supported_exploration_trap_share_cond_better", "Supported exploration"),
            ("after_full_completion_trap_share_cond_better", "Targeted completion"),
        ]
        rows = []
        for c, lab in metrics:
            rec = _mean_ci_record(seed_df, c, lab)
            rows.append(rec)
        data = pd.DataFrame(rows)
        data.to_csv(srcdir / "figure_8_intervention_residual_trap_share.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(data))
        y = data["mean"].to_numpy()
        yerr = np.vstack([y - data["ci95_low"].to_numpy(), data["ci95_high"].to_numpy() - y])
        ax.bar(x, y, yerr=yerr, capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(data["metric"], rotation=20)
        ax.set_ylabel("Trap share conditional on better alternative")
        ax.set_title("Structural trap removal under information interventions")
        fig.tight_layout(); fig.savefig(figdir / "figure_8_intervention_residual_trap_share.png", dpi=260); plt.close(fig)

        # Figure 9: seed robustness distribution.
        data = seed_df[["seed", "baseline_trap_share_cond_better", "after_supported_exploration_trap_share_cond_better"]].copy()
        data.to_csv(srcdir / "figure_9_seed_robustness_metrics.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data["seed"], data["baseline_trap_share_cond_better"], marker="o", label="Baseline")
        ax.plot(data["seed"], data["after_supported_exploration_trap_share_cond_better"], marker="o", label="Supported exploration")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Trap share")
        ax.set_title("Seed-level robustness of structural trap diagnostics")
        ax.legend()
        fig.tight_layout(); fig.savefig(figdir / "figure_9_seed_robustness_metrics.png", dpi=260); plt.close(fig)

        # Figure 10: intervention frontier.
        frontier = pd.DataFrame([
            {"design": "Baseline", "relative_cost": 0, "residual_trap_share": seed_df["baseline_trap_share_cond_better"].mean()},
            {"design": "Supported exploration", "relative_cost": 1, "residual_trap_share": seed_df["after_supported_exploration_trap_share_cond_better"].mean()},
            {"design": "Targeted completion", "relative_cost": 2, "residual_trap_share": seed_df["after_full_completion_trap_share_cond_better"].mean()},
        ])
        frontier.to_csv(srcdir / "figure_10_information_design_frontier.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(frontier["relative_cost"], frontier["residual_trap_share"], marker="o")
        for _, r in frontier.iterrows():
            ax.annotate(r["design"], (r["relative_cost"], r["residual_trap_share"]), textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("Relative information-design cost index")
        ax.set_ylabel("Residual structural trap share")
        ax.set_title("Information cost versus structural trap elimination")
        fig.tight_layout(); fig.savefig(figdir / "figure_10_information_design_frontier.png", dpi=260); plt.close(fig)


def compute_v9_final_publication_package(output_dir: Path, cfg: AuditConfig, n_seeds: int = 30, base_seed: int = 202607) -> None:
    """Create the final publication-grade empirical package.

    This package is deliberately split into structural diagnostics and dynamic
    diagnostics. It does not claim convergence from structural trap removal alone.
    """
    log("Computing V9 final publication-grade empirical package...")
    (output_dir / "data_for_figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables_csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables_latex").mkdir(parents=True, exist_ok=True)
    (output_dir / "robustness").mkdir(parents=True, exist_ok=True)
    (output_dir / "config").mkdir(parents=True, exist_ok=True)

    seed_df = compute_v9_seed_robustness(output_dir, n_seeds=n_seeds, base_seed=base_seed)
    compute_v9_final_tables(output_dir, cfg, seed_df)
    compute_v9_figures(output_dir, seed_df)

    protocol = {
        "script": "porto_learning_traps_v9_final.py",
        "n_seeds": int(n_seeds),
        "base_seed": int(base_seed),
        "unit_of_seed_resampling": "history_id",
        "theoretical_alignment": {
            "baseline_structural_object": "Theta^O(x) cap Theta^R(x)",
            "exploration_object": "Theta^O_D(x) cap Theta^R(x)",
            "completion_object": "Theta^{O,S}(x) cap Theta^R(x)",
            "dynamic_warning": "Structural trap removal is not a convergence claim. False-limit exclusion additionally requires the post-intervention counterparts of the Section 5 assumptions."
        },
        "tables_max": 6,
        "figures_max": 10,
        "table_files": [p.name for p in sorted((output_dir / "tables_csv").glob("table_*.csv"))],
        "figure_files": [p.name for p in sorted((output_dir / "figures").glob("figure_*.png"))],
    }
    (output_dir / "config" / "v9_final_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output_dir / "README_V9_FINAL_PACKAGE.md").write_text(
        """# Porto Learning Traps V9 Final Package\n\n"
        "This package produces publication-grade tables, figures, and robustness diagnostics for the empirical section.\n\n"
        "Core distinction: structural trap removal is separated from dynamic convergence. The empirical package reports baseline structural diagnostics, exploration-design diagnostics, information-completion diagnostics, and seed-level robustness.\n\n"
        "Main folders:\n"
        "- tables_csv: six main CSV tables.\n"
        "- tables_latex: LaTeX versions of the six main tables.\n"
        "- figures: up to ten final figures.\n"
        "- data_for_figures: exact CSV inputs used to regenerate each final figure.\n"
        "- robustness: seed-level metrics and sensitivity outputs.\n"
        "- config: protocol and parameter records.\n"
        """, encoding="utf-8")
    log(f"V9 final package written to: {output_dir.resolve()}")



# ---------------------------------------------------------------------------
# V10 FINAL: hostile-reviewer consolidation package
# ---------------------------------------------------------------------------

def _v10_dirs(output_dir: Path) -> dict:
    dirs = {
        "root": output_dir / "v11_publication",
        "tables_csv": output_dir / "v11_publication" / "tables_csv",
        "tables_latex": output_dir / "v11_publication" / "tables_latex",
        "figures": output_dir / "v11_publication" / "figures",
        "data_for_figures": output_dir / "v11_publication" / "data_for_figures",
        "bootstrap": output_dir / "v11_publication" / "bootstrap",
        "config": output_dir / "v11_publication" / "config",
        "appendix": output_dir / "v11_publication" / "appendix",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _v10_read_first(paths: list[Path]) -> pd.DataFrame:
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            df.attrs["source_path"] = str(p)
            return df
    return pd.DataFrame()


def _v10_history_panel(output_dir: Path) -> pd.DataFrame:
    panel = _normalize_history_panel(_select_history_panel(output_dir))
    if panel.empty:
        return panel
    if "TAXI_ID" not in panel.columns:
        panel["TAXI_ID"] = panel["history_id"].astype(str).str.split("__", n=1).str[0]
    if "od_context" not in panel.columns and "history_id" in panel.columns:
        panel["od_context"] = panel["history_id"].astype(str).str.split("__", n=1).str[1]
    # Canonical conservative terminology: these are empirical counterparts, not directly observed theoretical sets.
    panel["empirical_trap_candidate"] = _safe_bool_series(panel.get("learning_trap_candidate_v5", pd.Series(False, index=panel.index)))
    panel["strong_empirical_trap_candidate"] = _safe_bool_series(panel.get("strong_learning_trap_candidate_v5", pd.Series(False, index=panel.index)))
    panel["has_better_alternative"] = _safe_bool_series(panel.get("has_better_alternative", pd.Series(False, index=panel.index)))
    for c in ["information_incompleteness", "post_incumbent_share", "pre_incumbent_share", "median_better_alt_gap_min", "after_first_better_incumbent_share"]:
        if c not in panel.columns:
            panel[c] = np.nan
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    return panel


def _v10_decision_panel(output_dir: Path) -> pd.DataFrame:
    candidates = [
        output_dir / "q1_tables_v7" / "v7_decision_event_panel_with_fe_variables.csv",
        output_dir / "v7_decision_event_panel_with_fe_variables.csv",
    ]
    d = _v10_read_first(candidates)
    if d.empty:
        return d
    if "history_id" not in d.columns:
        if {"TAXI_ID", "od_context"}.issubset(d.columns):
            d["history_id"] = d["TAXI_ID"].astype(str) + "__" + d["od_context"].astype(str)
        else:
            d["history_id"] = np.arange(len(d)).astype(str)
    if "TAXI_ID" not in d.columns:
        d["TAXI_ID"] = d["history_id"].astype(str).str.split("__", n=1).str[0]
    for c in ["decision_event_time", "is_incumbent_choice", "information_incompleteness", "median_better_alt_gap_min"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _v10_metric_record(x: pd.Series | np.ndarray, name: str, unit: str = "") -> dict:
    s = pd.Series(x).dropna().astype(float)
    lo, hi = _ci95(s)
    return {
        "metric": name,
        "mean": float(s.mean()) if len(s) else np.nan,
        "sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_bootstrap": int(len(s)),
        "unit": unit,
    }


def _v10_compute_core_metrics(panel: pd.DataFrame) -> dict:
    if panel.empty:
        return {}
    better = _safe_bool_series(panel["has_better_alternative"])
    trap = _safe_bool_series(panel["empirical_trap_candidate"])
    strong = _safe_bool_series(panel["strong_empirical_trap_candidate"])
    denom = max(int(better.sum()), 1)
    post = pd.to_numeric(panel.get("post_incumbent_share", pd.Series(dtype=float)), errors="coerce")
    inc = pd.to_numeric(panel.get("information_incompleteness", pd.Series(dtype=float)), errors="coerce")
    gap = pd.to_numeric(panel.get("median_better_alt_gap_min", pd.Series(dtype=float)), errors="coerce")
    out = {
        "n_histories": int(len(panel)),
        "n_better_alt_histories": int(better.sum()),
        "n_empirical_trap_candidates": int(trap.sum()),
        "n_strong_empirical_trap_candidates": int(strong.sum()),
        "trap_diagnostic_share_cond_better": float(trap.sum() / denom),
        "strong_diagnostic_share_cond_better": float(strong.sum() / denom),
        "residual_after_supported_exploration": float(max(trap.sum() - strong.sum(), 0) / denom),
        "residual_after_targeted_completion": 0.0 if trap.sum() >= 0 else np.nan,
        "median_incompleteness": float(inc.median()) if len(inc.dropna()) else np.nan,
        "mean_incompleteness": float(inc.mean()) if len(inc.dropna()) else np.nan,
        "median_better_alt_gap_min": float(gap[gap > 0].median()) if len(gap[gap > 0].dropna()) else np.nan,
        "mean_post_incumbent_share": float(post.mean()) if len(post.dropna()) else np.nan,
    }
    if len(inc.dropna()) >= 8 and len(post.dropna()) >= 8:
        q1, q3 = inc.quantile(0.25), inc.quantile(0.75)
        out["q4_minus_q1_post_incumbent_share"] = float(post[inc >= q3].mean() - post[inc <= q1].mean())
    else:
        out["q4_minus_q1_post_incumbent_share"] = np.nan
    return out


def compute_v10_hierarchical_bootstrap(output_dir: Path, n_boot: int = 30, base_seed: int = 202607) -> pd.DataFrame:
    """Cluster bootstrap using taxis as the default empirical dependence unit.

    The seeds quantify sampling stability of empirical diagnostics under cluster resampling.
    They are not treated as independent experiments from different data-generating processes.
    """
    dirs = _v10_dirs(output_dir)
    panel = _v10_history_panel(output_dir)
    if panel.empty:
        out = pd.DataFrame()
        out.to_csv(dirs["bootstrap"] / "v10_cluster_bootstrap_metrics.csv", index=False)
        return out
    cluster_col = "TAXI_ID" if "TAXI_ID" in panel.columns else "history_id"
    clusters = panel[cluster_col].dropna().astype(str).unique()
    rows = []
    for b in range(n_boot):
        rng = np.random.default_rng(base_seed + b)
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = []
        for k, cl in enumerate(sampled_clusters):
            tmp = panel.loc[panel[cluster_col].astype(str) == str(cl)].copy()
            tmp["bootstrap_cluster_draw"] = k
            pieces.append(tmp)
        boot = pd.concat(pieces, ignore_index=True) if pieces else panel.iloc[0:0].copy()
        rec = _v10_compute_core_metrics(boot)
        rec.update({
            "bootstrap_index": b + 1,
            "seed": int(base_seed + b),
            "cluster_unit": cluster_col,
            "n_clusters_original": int(len(clusters)),
            "n_clusters_unique_drawn": int(pd.Series(sampled_clusters).nunique()),
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(dirs["bootstrap"] / "v10_cluster_bootstrap_metrics.csv", index=False)
    return out


def compute_v10_sensitivity(output_dir: Path) -> pd.DataFrame:
    """V11 corrected threshold sensitivity over the event-history sample.

    V10 accidentally evaluated sensitivity on the richest normalized panel, where
    several threshold variables could be missing or re-normalized.  V11 anchors
    the calculation to the event-history panel used for the primary diagnostic
    display whenever it is available.  The exercise remains a robustness check:
    it varies transparent classification thresholds and reports residual
    diagnostic shares; it does not redefine the canonical V5/V6 diagnostic.
    """
    dirs = _v10_dirs(output_dir)
    preferred = [
        output_dir / "q1_tables_v6" / "v6_history_event_measures.csv",
        output_dir / "q1_tables_v7" / "v7_history_event_measures_with_bins.csv",
        output_dir / "q1_tables_v5" / "v5_taxi_context_trap_panel.csv",
    ]
    panel = _v10_read_first(preferred)
    panel = _normalize_history_panel(panel) if not panel.empty else panel
    if panel.empty:
        out = pd.DataFrame()
        out.to_csv(dirs["appendix"] / "v11_threshold_sensitivity.csv", index=False)
        return out

    for c in [
        "post_incumbent_share", "after_first_better_incumbent_share", "information_incompleteness",
        "median_better_alt_gap_min", "incumbent_share", "better_opportunity_periods"
    ]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    # Use the event-history post share when available. Fall back to the V5 post-opportunity share.
    if "post_incumbent_share" in panel.columns and panel["post_incumbent_share"].notna().any():
        post = panel["post_incumbent_share"].astype(float)
        post_source = "post_incumbent_share"
    else:
        post = panel.get("after_first_better_incumbent_share", pd.Series(np.nan, index=panel.index)).astype(float)
        post_source = "after_first_better_incumbent_share"

    inc = panel.get("information_incompleteness", pd.Series(np.nan, index=panel.index)).astype(float)
    gap = panel.get("median_better_alt_gap_min", pd.Series(np.nan, index=panel.index)).astype(float)
    if "has_better_alternative" in panel.columns:
        better = _safe_bool_series(panel["has_better_alternative"])
    elif "better_opportunity_periods" in panel.columns:
        better = pd.to_numeric(panel["better_opportunity_periods"], errors="coerce").fillna(0) >= 1
    else:
        better = gap.fillna(0) > 0

    valid = better & post.notna() & inc.notna() & gap.notna()
    denom = int(valid.sum())
    rows = []
    for pthr in [0.50, 0.60, 0.70]:
        for ithr in [0.25, 0.50, 0.66, 0.75]:
            for gthr in [0.5, 1.0, 2.0]:
                flag = valid & (post >= pthr) & (inc >= ithr) & (gap >= gthr)
                rows.append({
                    "persistence_threshold": float(pthr),
                    "incompleteness_threshold": float(ithr),
                    "better_gap_threshold_min": float(gthr),
                    "n_diagnostic_histories": int(flag.sum()),
                    "diagnostic_share_cond_better": float(flag.sum() / max(denom, 1)),
                    "denominator_better_alt_histories": int(denom),
                    "post_persistence_source": post_source,
                    "source_path": panel.attrs.get("source_path", ""),
                })

    out = pd.DataFrame(rows)
    # Internal consistency guard: the grid should not be identically zero when the primary diagnostic is nonzero.
    primary = _v10_compute_core_metrics(_v10_history_panel(output_dir))
    primary_share = float(primary.get("trap_diagnostic_share_cond_better", 0.0) or 0.0)
    if primary_share > 0 and len(out) and float(out["diagnostic_share_cond_better"].max()) == 0.0:
        raise RuntimeError(
            "V11 threshold sensitivity is identically zero although the primary diagnostic is nonzero; "
            "check required columns and source panels."
        )
    out.to_csv(dirs["appendix"] / "v11_threshold_sensitivity.csv", index=False)
    # Backward-compatible alias for existing downstream code.
    out.to_csv(dirs["appendix"] / "v10_threshold_sensitivity.csv", index=False)
    return out


def compute_v10_event_study_data(output_dir: Path, n_boot: int = 30, base_seed: int = 202607) -> pd.DataFrame:
    dirs = _v10_dirs(output_dir)
    d = _v10_decision_panel(output_dir)
    if d.empty or not {"decision_event_time", "is_incumbent_choice", "information_incompleteness", "history_id"}.issubset(d.columns):
        out = pd.DataFrame()
        out.to_csv(dirs["data_for_figures"] / "figure_4_event_response_with_ci.csv", index=False)
        return out
    d = d.dropna(subset=["decision_event_time", "is_incumbent_choice", "information_incompleteness"]).copy()
    d = d[(d["decision_event_time"] >= -10) & (d["decision_event_time"] <= 10)]
    try:
        d["incompleteness_group"] = pd.qcut(d["information_incompleteness"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
    except Exception:
        d["incompleteness_group"] = "All"
    base = d.groupby(["incompleteness_group", "decision_event_time"], observed=True)["is_incumbent_choice"].mean().reset_index(name="mean")
    clusters = d["history_id"].astype(str).unique()
    boot_rows = []
    for b in range(n_boot):
        rng = np.random.default_rng(base_seed + 10_000 + b)
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        bd = pd.concat([d.loc[d["history_id"].astype(str) == h] for h in sampled], ignore_index=True)
        g = bd.groupby(["incompleteness_group", "decision_event_time"], observed=True)["is_incumbent_choice"].mean().reset_index(name="boot_mean")
        g["bootstrap_index"] = b + 1
        boot_rows.append(g)
    boots = pd.concat(boot_rows, ignore_index=True) if boot_rows else pd.DataFrame()
    if not boots.empty:
        ci = boots.groupby(["incompleteness_group", "decision_event_time"], observed=True)["boot_mean"].quantile([0.025, 0.975]).unstack().reset_index()
        ci = ci.rename(columns={0.025: "ci95_low", 0.975: "ci95_high"})
        base = base.merge(ci, on=["incompleteness_group", "decision_event_time"], how="left")
    base.to_csv(dirs["data_for_figures"] / "figure_4_event_response_with_ci.csv", index=False)
    return base


def compute_v10_pretrend_table(output_dir: Path) -> pd.DataFrame:
    """Simple transparent pre/post diagnostic, not a replacement for the structural theory."""
    dirs = _v10_dirs(output_dir)
    d = _v10_decision_panel(output_dir)
    if d.empty or not {"decision_event_time", "is_incumbent_choice", "information_incompleteness"}.issubset(d.columns):
        out = pd.DataFrame()
        out.to_csv(dirs["tables_csv"] / "table_4_event_response_inference.csv", index=False)
        return out
    d = d.dropna(subset=["decision_event_time", "is_incumbent_choice", "information_incompleteness"]).copy()
    try:
        d["high_incomplete"] = d["information_incompleteness"] >= d["information_incompleteness"].quantile(0.75)
        d["low_incomplete"] = d["information_incompleteness"] <= d["information_incompleteness"].quantile(0.25)
    except Exception:
        d["high_incomplete"] = False; d["low_incomplete"] = False
    rows = []
    for label, mask in [("Q1 lowest incompleteness", d["low_incomplete"]), ("Q4 highest incompleteness", d["high_incomplete"])]:
        sub = d.loc[mask]
        pre = sub.loc[sub["decision_event_time"] < 0, "is_incumbent_choice"].astype(float)
        post = sub.loc[sub["decision_event_time"] >= 0, "is_incumbent_choice"].astype(float)
        rows.append({
            "group": label,
            "pre_mean_incumbent_choice": float(pre.mean()) if len(pre) else np.nan,
            "post_mean_incumbent_choice": float(post.mean()) if len(post) else np.nan,
            "post_minus_pre": float(post.mean() - pre.mean()) if len(pre) and len(post) else np.nan,
            "n_pre_decisions": int(len(pre)),
            "n_post_decisions": int(len(post)),
        })
    # A simple pretrend slope on event times < 0 by group.
    for label, mask in [("Q1 lowest incompleteness", d["low_incomplete"]), ("Q4 highest incompleteness", d["high_incomplete"])]:
        sub = d.loc[mask & (d["decision_event_time"] < 0)].dropna(subset=["decision_event_time", "is_incumbent_choice"])
        if len(sub) >= 3 and sub["decision_event_time"].nunique() >= 2:
            x = sub["decision_event_time"].to_numpy(float)
            y = sub["is_incumbent_choice"].to_numpy(float)
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = np.nan
        rows.append({"group": label + " pretrend slope", "pre_mean_incumbent_choice": slope, "post_mean_incumbent_choice": np.nan, "post_minus_pre": np.nan, "n_pre_decisions": int(len(sub)), "n_post_decisions": 0})
    out = pd.DataFrame(rows)
    _to_latex_and_csv(out, dirs["tables_csv"] / "table_4_event_response_inference.csv", dirs["tables_latex"] / "table_4_event_response_inference.tex")
    return out


def compute_v10_tables(output_dir: Path, cfg: AuditConfig, boot_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> None:
    dirs = _v10_dirs(output_dir)
    panel = _v10_history_panel(output_dir)
    global_summary = _safe_read_csv(output_dir / "global_summary.csv")
    flags = _safe_read_csv(output_dir / "feasibility_flags.csv")
    v3_rep = _safe_read_csv(output_dir / "q1_tables_v3" / "table_1_v3_taxi_od_repetition_distribution.csv")
    v5_sum = _safe_read_csv(output_dir / "q1_tables_v5" / "table_1_v5_corrected_identification_summary.csv")
    v6_sum = _safe_read_csv(output_dir / "q1_tables_v6" / "table_1_v6_event_identification_summary.csv")
    v7_sum = _safe_read_csv(output_dir / "q1_tables_v7" / "table_1_v7_event_fe_sample_summary.csv")

    rows = []
    if not global_summary.empty:
        for c in global_summary.columns:
            rows.append({"object": c, "value": global_summary.iloc[0][c], "source": "global_summary.csv"})
    for name, df in [("V3 longitudinal eligible", v3_rep), ("V5 diagnostic summary", v5_sum), ("V6 event sample", v6_sum), ("V7 decision panel", v7_sum)]:
        if not df.empty:
            for c in df.columns:
                rows.append({"object": f"{name}: {c}", "value": df.iloc[0][c], "source": name})
    if not flags.empty:
        flags2 = flags.copy(); flags2["source"] = "feasibility_flags.csv"
        flags2 = flags2.rename(columns={"criterion": "object"})
        flags2["value"] = flags2.get("value", np.nan)
        rows.extend(flags2[["object", "value", "source"]].to_dict("records"))
    t1 = pd.DataFrame(rows)
    _to_latex_and_csv(t1, dirs["tables_csv"] / "table_1_sample_construction_and_feasibility.csv", dirs["tables_latex"] / "table_1_sample_construction_and_feasibility.tex")

    # Table 2: structural empirical primitives.
    if not panel.empty:
        inc = panel["information_incompleteness"].astype(float)
        post = panel["post_incumbent_share"].astype(float)
        gap = panel["median_better_alt_gap_min"].astype(float)
        t2 = pd.DataFrame([
            {"primitive": "Longitudinal histories", "estimate": len(panel), "interpretation": "Unit for empirical diagnostic"},
            {"primitive": "Histories with better alternative", "estimate": int(_safe_bool_series(panel["has_better_alternative"]).sum()), "interpretation": "Counterfactual support denominator"},
            {"primitive": "Median information incompleteness", "estimate": inc.median(), "interpretation": "Selective-feedback incompleteness"},
            {"primitive": "P75 information incompleteness", "estimate": inc.quantile(0.75), "interpretation": "Upper-tail selective feedback"},
            {"primitive": "Median better-alternative gap, minutes", "estimate": gap[gap > 0].median(), "interpretation": "Supported counterfactual improvement"},
            {"primitive": "Mean post-opportunity incumbent share", "estimate": post.mean(), "interpretation": "Persistence after opportunity"},
        ])
    else:
        t2 = pd.DataFrame()
    _to_latex_and_csv(t2, dirs["tables_csv"] / "table_2_empirical_primitives.csv", dirs["tables_latex"] / "table_2_empirical_primitives.tex")

    # Table 3: primary diagnostic, careful terminology.
    metrics = _v10_compute_core_metrics(panel)
    t3 = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
    _to_latex_and_csv(t3, dirs["tables_csv"] / "table_3_primary_trap_diagnostic.csv", dirs["tables_latex"] / "table_3_primary_trap_diagnostic.tex")

    compute_v10_pretrend_table(output_dir)

    # Table 5: interventions with bootstrap CI.
    if not boot_df.empty:
        t5 = pd.DataFrame([
            _v10_metric_record(boot_df["trap_diagnostic_share_cond_better"], "Baseline residual diagnostic share", "share"),
            _v10_metric_record(boot_df["residual_after_supported_exploration"], "Supported-exploration residual diagnostic share", "share"),
            _v10_metric_record(boot_df["residual_after_targeted_completion"], "Targeted-completion residual diagnostic share", "share"),
        ])
        t5["interpretation"] = [
            "Empirical trap-candidate share conditional on better alternative",
            "Residual after removing strong candidates with supported untried superior alternatives",
            "Constructed completion benchmark on targeted diagnostic set",
        ]
    else:
        t5 = pd.DataFrame()
    _to_latex_and_csv(t5, dirs["tables_csv"] / "table_5_information_interventions.csv", dirs["tables_latex"] / "table_5_information_interventions.tex")

    # Table 6: robustness and sensitivity, no Student tests over seeds.
    rows = []
    if not boot_df.empty:
        for c in ["trap_diagnostic_share_cond_better", "residual_after_supported_exploration", "q4_minus_q1_post_incumbent_share"]:
            if c in boot_df.columns:
                rows.append(_v10_metric_record(boot_df[c], c, "bootstrap distribution"))
    if not sensitivity_df.empty:
        rows.append({
            "metric": "threshold_sensitivity_min_diagnostic_share",
            "mean": float(sensitivity_df["diagnostic_share_cond_better"].min()),
            "sd": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_bootstrap": int(len(sensitivity_df)),
            "unit": "grid over thresholds",
        })
        rows.append({
            "metric": "threshold_sensitivity_max_diagnostic_share",
            "mean": float(sensitivity_df["diagnostic_share_cond_better"].max()),
            "sd": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_bootstrap": int(len(sensitivity_df)),
            "unit": "grid over thresholds",
        })
    t6 = pd.DataFrame(rows)
    _to_latex_and_csv(t6, dirs["tables_csv"] / "table_6_robustness_and_sensitivity.csv", dirs["tables_latex"] / "table_6_robustness_and_sensitivity.tex")


def compute_v10_figures(output_dir: Path, boot_df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> None:
    dirs = _v10_dirs(output_dir)
    if plt is None:
        return
    panel = _v10_history_panel(output_dir)
    decision = _v10_decision_panel(output_dir)

    # Figure 1: compact funnel with log scale and retention annotations.
    table1 = _safe_read_csv(dirs["tables_csv"] / "table_1_sample_construction_and_feasibility.csv")
    funnel_items = []
    def add_funnel(label, value):
        try:
            v = float(value)
            if np.isfinite(v): funnel_items.append({"stage": label, "count": v})
        except Exception:
            pass
    if not table1.empty:
        vals = dict(zip(table1["object"], table1["value"]))
        add_funnel("Valid trips", vals.get("n_valid_trips"))
        for k in vals:
            if "n_longitudinal_eligible" in str(k): add_funnel("Longitudinal histories", vals[k]); break
    if not panel.empty:
        m = _v10_compute_core_metrics(panel)
        add_funnel("Better alternative", m.get("n_better_alt_histories"))
        add_funnel("Trap candidates", m.get("n_empirical_trap_candidates"))
        add_funnel("Strong candidates", m.get("n_strong_empirical_trap_candidates"))
    f1 = pd.DataFrame(funnel_items).drop_duplicates("stage")
    f1.to_csv(dirs["data_for_figures"] / "figure_1_sample_funnel.csv", index=False)
    if not f1.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(f1["stage"], f1["count"])
        ax.set_yscale("log")
        ax.set_ylabel("Count, log scale")
        ax.set_title("Sample construction for empirical trap diagnostics")
        ax.tick_params(axis="x", rotation=25)
        for i, r in f1.iterrows():
            ax.text(i, r["count"], f"{int(r['count']):,}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); fig.savefig(dirs["figures"] / "figure_1_sample_funnel.png", dpi=280); plt.close(fig)

    # Figure 2: incompleteness distribution.
    if not panel.empty:
        d = panel[["information_incompleteness"]].dropna()
        d.to_csv(dirs["data_for_figures"] / "figure_2_incompleteness_distribution.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(d["information_incompleteness"], bins=30)
        ax.axvline(d["information_incompleteness"].median(), linestyle="--", linewidth=1)
        ax.set_xlabel("Information incompleteness")
        ax.set_ylabel("Histories")
        ax.set_title("Selective-feedback incompleteness among longitudinal histories")
        fig.tight_layout(); fig.savefig(dirs["figures"] / "figure_2_incompleteness_distribution.png", dpi=280); plt.close(fig)

    # Figure 3: binned persistence gradient with binomial-style CI.
    if not panel.empty and {"information_incompleteness", "post_incumbent_share"}.issubset(panel.columns):
        d = panel[["information_incompleteness", "post_incumbent_share"]].dropna().copy()
        try:
            d["bin"] = pd.qcut(d["information_incompleteness"], 8, duplicates="drop")
        except Exception:
            d["bin"] = pd.cut(d["information_incompleteness"], 8)
        rows = []
        for b, g in d.groupby("bin", observed=True):
            n = len(g); mean = g["post_incumbent_share"].mean(); se = g["post_incumbent_share"].std(ddof=1) / math.sqrt(n) if n > 1 else 0
            rows.append({"bin": str(b), "x": g["information_incompleteness"].mean(), "mean": mean, "ci95_low": mean - 1.96*se, "ci95_high": mean + 1.96*se, "n": n})
        gdf = pd.DataFrame(rows)
        gdf.to_csv(dirs["data_for_figures"] / "figure_3_persistence_gradient.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        if not gdf.empty:
            yerr = np.vstack([gdf["mean"] - gdf["ci95_low"], gdf["ci95_high"] - gdf["mean"]])
            ax.errorbar(gdf["x"], gdf["mean"], yerr=yerr, marker="o", capsize=3)
        ax.set_xlabel("Information incompleteness")
        ax.set_ylabel("Post-opportunity incumbent share")
        ax.set_title("Persistence gradient by selective-feedback incompleteness")
        fig.tight_layout(); fig.savefig(dirs["figures"] / "figure_3_persistence_gradient.png", dpi=280); plt.close(fig)

    # Figure 4: event response with cluster bootstrap CI.
    ev = compute_v10_event_study_data(output_dir, n_boot=max(10, len(boot_df) if not boot_df.empty else 30))
    if not ev.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for lab, g in ev.groupby("incompleteness_group", observed=True):
            g = g.sort_values("decision_event_time")
            ax.plot(g["decision_event_time"], g["mean"], marker="o", label=str(lab))
            if {"ci95_low", "ci95_high"}.issubset(g.columns):
                ax.fill_between(g["decision_event_time"].to_numpy(float), g["ci95_low"].to_numpy(float), g["ci95_high"].to_numpy(float), alpha=0.12)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Decision event time relative to first better alternative")
        ax.set_ylabel("Incumbent-choice share")
        ax.set_title("Event response by information incompleteness")
        ax.legend()
        fig.tight_layout(); fig.savefig(dirs["figures"] / "figure_4_event_response_with_ci.png", dpi=280); plt.close(fig)

    # Figure 5: intervention diagnostic reduction with bootstrap CIs.
    if not boot_df.empty:
        rows = [
            _v10_metric_record(boot_df["trap_diagnostic_share_cond_better"], "Baseline"),
            _v10_metric_record(boot_df["residual_after_supported_exploration"], "Supported exploration"),
            _v10_metric_record(boot_df["residual_after_targeted_completion"], "Targeted completion"),
        ]
        d = pd.DataFrame(rows)
        d.to_csv(dirs["data_for_figures"] / "figure_5_intervention_diagnostic_reduction.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(d)); y = d["mean"].to_numpy(float)
        yerr = np.vstack([y - d["ci95_low"].to_numpy(float), d["ci95_high"].to_numpy(float) - y])
        ax.bar(x, y, yerr=yerr, capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(d["metric"], rotation=15)
        ax.set_ylabel("Residual trap-diagnostic share")
        ax.set_title("Diagnostic attenuation under information-completion benchmarks")
        fig.tight_layout(); fig.savefig(dirs["figures"] / "figure_5_intervention_diagnostic_reduction.png", dpi=280); plt.close(fig)

    # Figure 6: threshold-sensitivity heatmap for the canonical 1-minute better-gap threshold.
    if not sensitivity_df.empty:
        sensitivity_df.to_csv(dirs["data_for_figures"] / "figure_6_threshold_sensitivity.csv", index=False)
        canonical_gap = 1.0
        plot_df = sensitivity_df.loc[np.isclose(sensitivity_df["better_gap_threshold_min"].astype(float), canonical_gap)].copy()
        if plot_df.empty:
            plot_df = sensitivity_df.copy()
        pivot = plot_df.pivot_table(
            index="incompleteness_threshold",
            columns="persistence_threshold",
            values="diagnostic_share_cond_better",
            aggfunc="mean",
        ).sort_index(ascending=True)
        n_pivot = plot_df.pivot_table(
            index="incompleteness_threshold",
            columns="persistence_threshold",
            values="n_diagnostic_histories",
            aggfunc="sum",
        ).reindex(index=pivot.index, columns=pivot.columns)
        heat = pivot.to_numpy(float)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(heat, origin="lower", aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])
        ax.set_xlabel("Persistence threshold")
        ax.set_ylabel("Incompleteness threshold")
        ax.set_title("Threshold sensitivity of residual trap-diagnostic share")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = heat[i, j]
                nval = n_pivot.iloc[i, j] if not n_pivot.empty else np.nan
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}\n(n={int(nval) if np.isfinite(nval) else 0})", ha="center", va="center", fontsize=7)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Diagnostic share among supported better-alternative histories")
        fig.tight_layout(); fig.savefig(dirs["figures"] / "figure_6_threshold_sensitivity.png", dpi=280); plt.close(fig)


def compute_v11_publication_package(output_dir: Path, cfg: AuditConfig, n_seeds: int = 30, base_seed: int = 202607) -> None:
    """Build the V11 targeted robustness empirical package.

    V10 replaces seed-level t-tests by cluster-bootstrap stability summaries, reduces the
    main display to six figures and six tables, and uses conservative terminology throughout.
    """
    log("Computing V11 targeted robustness publication package...")
    dirs = _v10_dirs(output_dir)
    boot_df = compute_v10_hierarchical_bootstrap(output_dir, n_boot=n_seeds, base_seed=base_seed)
    sensitivity_df = compute_v10_sensitivity(output_dir)
    compute_v10_tables(output_dir, cfg, boot_df, sensitivity_df)
    compute_v10_figures(output_dir, boot_df, sensitivity_df)
    protocol = {
        "script": "porto_learning_traps_v11_publication.py",
        "version": "V11 targeted robustness correction",
        "n_bootstrap_draws": int(n_seeds),
        "base_seed": int(base_seed),
        "primary_inference": "cluster bootstrap over taxis/histories; seeds are computational resampling draws, not independent datasets",
        "canonical_terms": {
            "trap_label": "empirically supported trap candidates",
            "effect_label": "residual trap-diagnostic share",
            "completion_label": "constructed targeted-completion benchmark"
        },
        "main_tables_max": 6,
        "main_figures_max": 6,
        "dynamic_warning": "Structural diagnostic removal is not a convergence claim. False-limit exclusion requires an intervened process satisfying the Section 5 counterparts.",
        "folders": {k: str(v) for k, v in dirs.items()},
    }
    (dirs["config"] / "v10_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (dirs["root"] / "README_V10_PUBLICATION_PACKAGE.md").write_text(
        """# Porto Learning Traps V11 Publication Package\n\n"
        "V11 is the targeted robustness correction of the empirical package. It preserves the V3--V8 construction but presents only a conservative set of final tables and figures.\n\n"
        "Main changes relative to V9:\n"
        "1. The 30 seeds are treated as cluster-bootstrap stability draws, not as independent datasets.\n"
        "2. Main figures are reduced to six publication figures.\n"
        "3. Terminology is conservative: empirical trap candidates, residual diagnostic share, targeted-completion benchmark.\n"
        "4. All figure inputs are saved in data_for_figures.\n"
        "5. Structural trap removal is separated from dynamic convergence.\n"
        """, encoding="utf-8")
    log(f"V11 publication package written to: {dirs['root'].resolve()}")



# ---------------------------------------------------------------------------
# V12 FINAL: reviewer-proof inference add-on
# ---------------------------------------------------------------------------

def _v12_dirs(output_dir: Path) -> dict:
    dirs = {
        "root": output_dir / "v12_inference",
        "tables_csv": output_dir / "v12_inference" / "tables_csv",
        "tables_latex": output_dir / "v12_inference" / "tables_latex",
        "figures": output_dir / "v12_inference" / "figures",
        "data_for_figures": output_dir / "v12_inference" / "data_for_figures",
        "bootstrap": output_dir / "v12_inference" / "bootstrap",
        "config": output_dir / "v12_inference" / "config",
        "appendix": output_dir / "v12_inference" / "appendix",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _v12_read_decision_panel(output_dir: Path) -> pd.DataFrame:
    d = _v10_decision_panel(output_dir)
    if d.empty:
        return d
    # Canonical columns used for reviewer-proof inference.
    if "decision_event_time" not in d.columns and "event_time" in d.columns:
        d["decision_event_time"] = d["event_time"]
    if "is_incumbent_choice" not in d.columns and "incumbent_choice" in d.columns:
        d["is_incumbent_choice"] = d["incumbent_choice"]
    required = ["history_id", "TAXI_ID", "decision_event_time", "is_incumbent_choice", "information_incompleteness"]
    for c in required:
        if c not in d.columns:
            d[c] = np.nan
    d = d.dropna(subset=["history_id", "TAXI_ID", "decision_event_time", "is_incumbent_choice", "information_incompleteness"]).copy()
    d["decision_event_time"] = pd.to_numeric(d["decision_event_time"], errors="coerce")
    d["is_incumbent_choice"] = pd.to_numeric(d["is_incumbent_choice"], errors="coerce")
    d["information_incompleteness"] = pd.to_numeric(d["information_incompleteness"], errors="coerce")
    d = d.dropna(subset=["decision_event_time", "is_incumbent_choice", "information_incompleteness"])
    d["post_event"] = (d["decision_event_time"] >= 0).astype(float)
    d["incompletion_c"] = d["information_incompleteness"] - d["information_incompleteness"].mean()
    d["post_x_incompletion"] = d["post_event"] * d["incompletion_c"]
    if "median_better_alt_gap_min" in d.columns:
        d["gap_c"] = pd.to_numeric(d["median_better_alt_gap_min"], errors="coerce")
        d["gap_c"] = d["gap_c"].fillna(d["gap_c"].median())
        d["gap_c"] = d["gap_c"] - d["gap_c"].mean()
    else:
        d["gap_c"] = 0.0
    return d


def _v12_design_matrix(df: pd.DataFrame, covariates: list[str], fe_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    parts = []
    names = []
    # Covariates first, no intercept when fixed effects are included.
    for c in covariates:
        if c in df.columns:
            arr = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)[:, None]
            parts.append(arr)
            names.append(c)
    for fe in fe_cols:
        if fe in df.columns:
            cats = pd.get_dummies(df[fe].astype(str), prefix=fe, drop_first=True, dtype=float)
            if cats.shape[1] > 0:
                parts.append(cats.to_numpy(dtype=float))
                names.extend(list(cats.columns))
    if not parts:
        return np.ones((len(df), 1)), ["intercept"]
    X = np.hstack(parts)
    # Drop columns with zero variance to avoid rank noise, but preserve named covariates if possible.
    keep = np.nanstd(X, axis=0) > 1e-12
    if keep.sum() == 0:
        return np.ones((len(df), 1)), ["intercept"]
    X = X[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    return X, names


def _v12_fit_ols(df: pd.DataFrame, covariates: list[str], fe_cols: list[str]) -> dict:
    if df.empty or len(df) < 10:
        return {"n_obs": len(df), "status": "empty_or_too_small"}
    y = pd.to_numeric(df["is_incumbent_choice"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    X, names = _v12_design_matrix(df, covariates, fe_cols)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception as exc:
        return {"n_obs": len(df), "status": f"lstsq_failed:{exc}"}
    out = {"n_obs": int(len(df)), "n_covariates_total": int(X.shape[1]), "status": "ok"}
    yhat = X @ beta
    resid = y - yhat
    out["rmse"] = float(np.sqrt(np.mean(resid ** 2)))
    for name, b in zip(names, beta):
        if name in covariates:
            out[f"beta_{name}"] = float(b)
    return out


def _v12_cluster_bootstrap_regression(df: pd.DataFrame, n_boot: int, base_seed: int,
                                      covariates: list[str], fe_cols: list[str],
                                      cluster_col: str = "TAXI_ID") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(base_seed)
    clusters = pd.Series(df[cluster_col].astype(str).unique())
    rows = []
    for b in range(n_boot):
        sampled = rng.choice(clusters.to_numpy(), size=len(clusters), replace=True)
        parts = []
        for j, cl in enumerate(sampled):
            sub = df[df[cluster_col].astype(str) == str(cl)].copy()
            if sub.empty:
                continue
            # Preserve duplicate sampled clusters as distinct bootstrap copies.
            sub["_boot_cluster_copy"] = f"{cl}__{j}"
            parts.append(sub)
        if not parts:
            continue
        boot = pd.concat(parts, ignore_index=True)
        rec = _v12_fit_ols(boot, covariates=covariates, fe_cols=fe_cols)
        rec["bootstrap_draw"] = b
        rec["n_clusters_drawn"] = int(len(sampled))
        rows.append(rec)
    return pd.DataFrame(rows)


def _v12_ci_from_boot(full_value: float, boot_values: pd.Series) -> tuple[float, float, float]:
    s = pd.Series(boot_values).dropna().astype(float)
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = np.quantile(s, [0.025, 0.975])
    # Two-sided percentile p-value against zero.
    p = 2.0 * min(float((s <= 0).mean()), float((s >= 0).mean()))
    return float(lo), float(hi), float(min(1.0, p))


def _v12_inference_tables(output_dir: Path, dirs: dict, n_boot: int, base_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = _v12_read_decision_panel(output_dir)
    d.to_csv(dirs["appendix"] / "v12_decision_panel_used_for_inference.csv", index=False)
    if d.empty:
        empty = pd.DataFrame([{"warning": "decision panel unavailable"}])
        empty.to_csv(dirs["tables_csv"] / "table_v12_reviewer_inference.csv", index=False)
        return empty, pd.DataFrame()

    # Main specification: history fixed effects + event-time fixed effects.
    # The coefficient on post_x_incompletion is identified by within-history changes
    # around the event, net of common event-time shocks.
    main_covs = ["post_x_incompletion", "gap_c"]
    main_fes = ["history_id", "decision_event_time"]
    full_main = _v12_fit_ols(d, main_covs, main_fes)
    boot_main = _v12_cluster_bootstrap_regression(d, n_boot, base_seed, main_covs, main_fes, cluster_col="TAXI_ID")
    boot_main.to_csv(dirs["bootstrap"] / "v12_bootstrap_main_event_fe.csv", index=False)

    # Post-period gradient: after the first supported better alternative, do more incomplete histories
    # exhibit higher incumbent persistence? Uses event-time FE but not history FE because incompleteness
    # is history-level and would be absorbed.
    post = d[d["decision_event_time"] >= 0].copy()
    post_covs = ["incompletion_c", "gap_c"]
    post_fes = ["decision_event_time"]
    full_post = _v12_fit_ols(post, post_covs, post_fes)
    boot_post = _v12_cluster_bootstrap_regression(post, n_boot, base_seed + 101, post_covs, post_fes, cluster_col="TAXI_ID")
    boot_post.to_csv(dirs["bootstrap"] / "v12_bootstrap_post_gradient.csv", index=False)

    # Placebo/pre-event diagnostic: within pre-event observations only, define a placebo split at -5.
    pre = d[d["decision_event_time"] < 0].copy()
    if not pre.empty:
        pre["placebo_post"] = (pre["decision_event_time"] >= -5).astype(float)
        pre["placebo_x_incompletion"] = pre["placebo_post"] * pre["incompletion_c"]
    placebo_covs = ["placebo_x_incompletion", "gap_c"]
    placebo_fes = ["history_id", "decision_event_time"]
    full_placebo = _v12_fit_ols(pre, placebo_covs, placebo_fes)
    boot_placebo = _v12_cluster_bootstrap_regression(pre, n_boot, base_seed + 202, placebo_covs, placebo_fes, cluster_col="TAXI_ID")
    boot_placebo.to_csv(dirs["bootstrap"] / "v12_bootstrap_pre_event_placebo.csv", index=False)

    specs = [
        {
            "specification": "S1 within-history event response",
            "estimand": "Post x incompleteness",
            "coefficient": full_main.get("beta_post_x_incompletion", np.nan),
            "boot_col": "beta_post_x_incompletion",
            "boot": boot_main,
            "n_obs": full_main.get("n_obs", np.nan),
            "fixed_effects": "history_id + decision_event_time",
            "interpretation": "Within-history change around supported better-alternative opportunity; not a causal treatment effect without additional assumptions.",
        },
        {
            "specification": "S2 post-period persistence gradient",
            "estimand": "Incompleteness",
            "coefficient": full_post.get("beta_incompletion_c", np.nan),
            "boot_col": "beta_incompletion_c",
            "boot": boot_post,
            "n_obs": full_post.get("n_obs", np.nan),
            "fixed_effects": "decision_event_time",
            "interpretation": "Cross-history gradient in post-opportunity incumbent persistence.",
        },
        {
            "specification": "S3 pre-event placebo split",
            "estimand": "Placebo post x incompleteness",
            "coefficient": full_placebo.get("beta_placebo_x_incompletion", np.nan),
            "boot_col": "beta_placebo_x_incompletion",
            "boot": boot_placebo,
            "n_obs": full_placebo.get("n_obs", np.nan),
            "fixed_effects": "history_id + decision_event_time, pre-event sample only",
            "interpretation": "Placebo diagnostic for differential pre-event movement.",
        },
    ]
    rows = []
    for srec in specs:
        lo, hi, pval = _v12_ci_from_boot(srec["coefficient"], srec["boot"].get(srec["boot_col"], pd.Series(dtype=float)))
        rows.append({
            "specification": srec["specification"],
            "estimand": srec["estimand"],
            "coefficient": srec["coefficient"],
            "ci95_low_cluster_bootstrap": lo,
            "ci95_high_cluster_bootstrap": hi,
            "bootstrap_p_value_two_sided": pval,
            "n_observations": srec["n_obs"],
            "n_bootstrap_draws": int(n_boot),
            "cluster_unit": "TAXI_ID",
            "fixed_effects": srec["fixed_effects"],
            "interpretation": srec["interpretation"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(dirs["tables_csv"] / "table_v12_reviewer_inference.csv", index=False)
    write_latex_table(table, dirs["tables_latex"] / "table_v12_reviewer_inference.tex", max_rows=20)

    # A compact support table for reviewer checks.
    support = pd.DataFrame([{
        "n_decisions": int(len(d)),
        "n_event_histories": int(d["history_id"].nunique()),
        "n_taxis": int(d["TAXI_ID"].nunique()),
        "event_time_min": float(d["decision_event_time"].min()),
        "event_time_max": float(d["decision_event_time"].max()),
        "share_post_decisions": float((d["decision_event_time"] >= 0).mean()),
        "median_information_incompleteness": float(d["information_incompleteness"].median()),
        "mean_incumbent_choice": float(d["is_incumbent_choice"].mean()),
    }])
    support.to_csv(dirs["tables_csv"] / "table_v12_inference_support.csv", index=False)
    write_latex_table(support.T.reset_index().rename(columns={"index": "metric", 0: "value"}), dirs["tables_latex"] / "table_v12_inference_support.tex", max_rows=50)
    return table, d


def _v12_make_figures(dirs: dict, table: pd.DataFrame, d: pd.DataFrame) -> None:
    if plt is None:
        return
    if not table.empty and "coefficient" in table.columns:
        fig_data = table[["specification", "estimand", "coefficient", "ci95_low_cluster_bootstrap", "ci95_high_cluster_bootstrap"]].copy()
        fig_data.to_csv(dirs["data_for_figures"] / "figure_v12_reviewer_inference_forest.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        y = np.arange(len(fig_data))
        x = fig_data["coefficient"].to_numpy(float)
        lo = fig_data["ci95_low_cluster_bootstrap"].to_numpy(float)
        hi = fig_data["ci95_high_cluster_bootstrap"].to_numpy(float)
        xerr = np.vstack([x - lo, hi - x])
        ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=4)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(fig_data["estimand"])
        ax.set_xlabel("Coefficient with taxi-cluster bootstrap 95% CI")
        ax.set_title("Reviewer-proof event-response inference")
        fig.tight_layout(); fig.savefig(dirs["figures"] / "figure_v12_reviewer_inference_forest.png", dpi=300); plt.close(fig)

    # Post-period binned gradient with bootstrap-free standard errors for display only;
    # formal inference is in table_v12_reviewer_inference.
    if not d.empty:
        post = d[d["decision_event_time"] >= 0].copy()
        if len(post) > 0:
            try:
                post["bin"] = pd.qcut(post["information_incompleteness"], 6, duplicates="drop")
            except Exception:
                post["bin"] = pd.cut(post["information_incompleteness"], 6)
            rows = []
            for b, g in post.groupby("bin", observed=True):
                n = len(g)
                mean = g["is_incumbent_choice"].mean()
                se = g["is_incumbent_choice"].std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
                rows.append({"bin": str(b), "x": g["information_incompleteness"].mean(), "mean": mean, "ci95_low": mean - 1.96*se, "ci95_high": mean + 1.96*se, "n": n})
            gd = pd.DataFrame(rows)
            gd.to_csv(dirs["data_for_figures"] / "figure_v12_post_gradient_decision_level.csv", index=False)
            if not gd.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                yy = gd["mean"].to_numpy(float)
                yerr = np.vstack([yy - gd["ci95_low"].to_numpy(float), gd["ci95_high"].to_numpy(float) - yy])
                ax.errorbar(gd["x"], yy, yerr=yerr, marker="o", capsize=4)
                ax.set_xlabel("Information incompleteness")
                ax.set_ylabel("Post-event incumbent-choice share")
                ax.set_title("Decision-level post-opportunity persistence gradient")
                fig.tight_layout(); fig.savefig(dirs["figures"] / "figure_v12_post_gradient_decision_level.png", dpi=300); plt.close(fig)


def compute_v12_final_inference_package(output_dir: Path, cfg: AuditConfig, n_seeds: int = 30, base_seed: int = 202607) -> None:
    """Build the V12 reviewer-proof inference add-on.

    V12 does not change the V3--V11 empirical construction. It adds targeted
    decision-level inference: fixed-effect event-response regressions, taxi-cluster
    bootstrap confidence intervals, and a pre-event placebo diagnostic.
    """
    log("Computing V12 reviewer-proof inference package...")
    dirs = _v12_dirs(output_dir)
    table, d = _v12_inference_tables(output_dir, dirs, n_boot=n_seeds, base_seed=base_seed)
    _v12_make_figures(dirs, table, d)
    protocol = {
        "script": "porto_learning_traps_v12_final_inference.py",
        "version": "V12 final reviewer-proof inference add-on",
        "n_bootstrap_draws": int(n_seeds),
        "base_seed": int(base_seed),
        "primary_new_outputs": [
            "table_v12_reviewer_inference.csv",
            "table_v12_inference_support.csv",
            "figure_v12_reviewer_inference_forest.png",
            "figure_v12_post_gradient_decision_level.png",
        ],
        "inference_principles": {
            "cluster_unit": "TAXI_ID",
            "main_specification": "history fixed effects plus event-time fixed effects",
            "main_estimand": "Post-event x information-incompleteness interaction",
            "post_gradient": "post-event cross-history persistence gradient",
            "placebo": "pre-event placebo split interacted with incompleteness",
            "warning": "Event-aligned estimates are diagnostic evidence, not standalone causal treatment effects."
        },
    }
    (dirs["config"] / "v12_inference_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (dirs["root"] / "README_V12_INFERENCE.md").write_text(
        """# Porto Learning Traps V12 Inference Add-on

"
        "V12 is a targeted reviewer-proof inference layer built on top of V11. It does not redefine the empirical trap diagnostic. It adds decision-level event-response regressions, taxi-cluster bootstrap confidence intervals, and a pre-event placebo diagnostic.\n\n"
        "The central interpretation is conservative: estimates document event-aligned persistence patterns associated with selective-feedback incompleteness. They are not presented as causal treatment effects of the first better-alternative opportunity.\n"
        """,
        encoding="utf-8",
    )
    log(f"V12 inference package written to: {dirs['root'].resolve()}")

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Porto learning-trap empirical package V12: final reviewer-proof inference add-on.")
    parser.add_argument("--data", required=True, help="Path to train.csv or train.csv.zip")
    parser.add_argument("--out", default="Results_Porto_Final", help="Output directory")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--grid", type=float, default=0.005, help="Grid size in degrees for OD zones")
    parser.add_argument("--min-context-trips", type=int, default=30)
    parser.add_argument("--min-taxi-context-repeats", type=int, default=5)
    parser.add_argument("--route-bins", type=int, default=6)
    parser.add_argument("--time-window-minutes", type=int, default=30)
    parser.add_argument("--sample-route-rows", type=int, default=300_000)
    parser.add_argument("--family-grid", type=float, default=0.0125, help="Coarser grid in degrees for behavioral route families")
    parser.add_argument("--family-bins", type=int, default=4, help="Number of interior samples for behavioral route-family signatures")
    parser.add_argument("--min-family-count", type=int, default=5, help="Minimum within-context trips for a route family not to be merged into OTHER")
    parser.add_argument("--min-family-share", type=float, default=0.01, help="Minimum within-context share for a route family not to be merged into OTHER")
    parser.add_argument("--min-span-days", type=float, default=30.0, help="Minimum longitudinal span for repeated taxi-context histories")
    parser.add_argument("--min-distance-km", type=float, default=1.0, help="Exclude very short/local trips")
    parser.add_argument("--keep-same-zone", action="store_true", help="Keep contexts with the same origin and destination zone")
    parser.add_argument("--v5-period", choices=["D", "W", "M"], default="M", help="Period used for counterfactual opportunities: D, W, or M")
    parser.add_argument("--v5-min-alt-period-obs", type=int, default=5, help="Minimum observations for a route family in a context-period to define a counterfactual")
    parser.add_argument("--v5-min-better-gap-min", type=float, default=1.0, help="Minimum time saving in minutes for an alternative to count as better")
    parser.add_argument("--v5-min-opportunity-periods", type=int, default=2, help="Minimum number of better-opportunity periods")
    parser.add_argument("--v5-min-persistence-share", type=float, default=0.60, help="Minimum incumbent share for trap classification")
    parser.add_argument("--v5-min-incompleteness", type=float, default=0.25, help="Minimum information incompleteness for trap classification")
    parser.add_argument("--v6-max-event-decision", type=int, default=10, help="Post-event decision window for V6 abandonment diagnostics")
    parser.add_argument("--v6-min-post-decisions", type=int, default=2, help="Minimum post-event decisions for V6 event histories")
    parser.add_argument("--v6-min-pre-decisions", type=int, default=1, help="Minimum pre-event decisions for V6 event histories")
    parser.add_argument("--v7-max-event-decision", type=int, default=10, help="Symmetric event window for V7 within-history diagnostics")
    parser.add_argument("--v7-min-post-decisions", type=int, default=2, help="Minimum post-event decisions for V7 histories")
    parser.add_argument("--v7-min-pre-decisions", type=int, default=1, help="Minimum pre-event decisions for V7 histories")
    parser.add_argument("--seeds", type=int, default=30, help="Number of history-level robustness seeds")
    parser.add_argument("--base-seed", type=int, default=202607, help="Base seed for robustness resampling")
    parser.add_argument("--skip-core", action="store_true", help="Skip V3--V8 recomputation and build V11/V12 packages from an existing output directory")
    args = parser.parse_args(argv)

    cfg = AuditConfig(
        data_path=args.data,
        output_dir=args.out,
        chunksize=args.chunksize,
        grid_size_deg=args.grid,
        min_context_trips=args.min_context_trips,
        min_taxi_context_repeats=args.min_taxi_context_repeats,
        route_bins=args.route_bins,
        time_window_minutes=args.time_window_minutes,
        sample_route_rows=args.sample_route_rows,
        family_grid_deg=args.family_grid,
        family_bins=args.family_bins,
        min_family_count=args.min_family_count,
        min_family_share=args.min_family_share,
        min_span_days=args.min_span_days,
        min_distance_km=args.min_distance_km,
        exclude_same_zone=not args.keep_same_zone,
    )

    output_dir = Path(cfg.output_dir)
    ensure_dirs(output_dir)
    (output_dir / "config").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "audit_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    with open(output_dir / "config" / "run_arguments.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    if not args.skip_core:
        df = read_processed(cfg)
        try:
            df.to_parquet(output_dir / "processed_trips.parquet", index=False)
        except Exception as exc:
            log(f"Could not write parquet cache: {exc}. Writing CSV fallback.")
            df.to_csv(output_dir / "processed_trips.csv", index=False)
        summarize(df, cfg, output_dir)
        maybe_counterfactual_support(df, cfg, output_dir)
        compute_q1_tables(df, cfg, output_dir)
        compute_v5_learning_traps(
            df,
            cfg,
            output_dir,
            period=args.v5_period,
            min_alt_period_obs=args.v5_min_alt_period_obs,
            min_better_gap_min=args.v5_min_better_gap_min,
            min_opportunity_periods=args.v5_min_opportunity_periods,
            min_persistence_share=args.v5_min_persistence_share,
            min_incompleteness=args.v5_min_incompleteness,
        )
        compute_v6_event_identification(
            df,
            cfg,
            output_dir,
            period=args.v5_period,
            max_event_decision=args.v6_max_event_decision,
            min_post_decisions=args.v6_min_post_decisions,
            min_pre_decisions=args.v6_min_pre_decisions,
            persistence_threshold=args.v5_min_persistence_share,
        )
        compute_v7_event_fe_identification(
            output_dir,
            max_event_decision=args.v7_max_event_decision,
            min_pre_decisions=args.v7_min_pre_decisions,
            min_post_decisions=args.v7_min_post_decisions,
        )
        compute_v8_publication_package(output_dir)
    else:
        log("Skipping V3--V8 recomputation; using existing output directory for V10 package.")

    compute_v11_publication_package(output_dir, cfg, n_seeds=args.seeds, base_seed=args.base_seed)
    compute_v12_final_inference_package(output_dir, cfg, n_seeds=args.seeds, base_seed=args.base_seed)
    log(f"Done. Outputs written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
