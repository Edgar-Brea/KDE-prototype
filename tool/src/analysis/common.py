"""Shared settings and loaders for the export, Figure 1 and scoring scripts."""
import os

import pandas as pd
import yaml

from src.validate import TYPES  # noqa: F401  (band order, bottom to top)

RETAINED, LONG_TEXT = "retained", "excluded_long_text"


def load_config(path="config.yaml"):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    return cfg, cfg["analysis"]


def export_paths(acfg):
    d = acfg["exports_dir"]
    return {"components": os.path.join(d, "components.parquet"),
            "components_csv": os.path.join(d, "components.csv.gz"),
            "index": os.path.join(d, "abstracts_index.csv")}


def read_exports(acfg):
    """Components and abstract index as written by export_components (parquet, or the gzipped CSV fallback)."""
    p = export_paths(acfg)
    comps = pd.read_parquet(p["components"]) if os.path.exists(p["components"]) else pd.read_csv(p["components_csv"])
    index = pd.read_csv(p["index"])
    for df in (comps, index):
        df["year"] = df["year"].astype("Int64")
    return comps, index
