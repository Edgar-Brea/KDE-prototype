"""Export the full-run components and an index of processed abstracts.

  python -m src.analysis.export_components [--config config.yaml]

Reads <run_dir>/outputs.jsonl (or outputs.jsonl.gz), the corpus (for year, period, search strand,
title and abstract length) and <run_dir>/details.jsonl (for had_retry). details.jsonl is local only; without it,
had_retry is taken from the existing exports/abstracts_index.csv.

Writes exports/components.parquet (one row per kept component), exports/abstracts_index.csv
(one row per processed abstract) and exports/long_text_report.md. No abstract text is exported.
An abstract longer than analysis.long_text_chars gets status excluded_long_text; all others are retained.
"""
import argparse
import gzip
import json
import os
import sys

import pandas as pd

from src.analysis.common import LONG_TEXT, RETAINED, export_paths, load_config
from src.run import load_corpus

MAX_MB = 50


def read_outputs(run_dir):
    plain, gz = os.path.join(run_dir, "outputs.jsonl"), os.path.join(run_dir, "outputs.jsonl.gz")
    f = open(plain, encoding="utf-8") if os.path.exists(plain) else gzip.open(gz, "rt", encoding="utf-8")
    with f:
        return [json.loads(l) for l in f if l.strip()]


def read_retries(run_dir, index_path):
    path = os.path.join(run_dir, "details.jsonl")
    if os.path.exists(path):
        retried = {}
        with open(path, encoding="utf-8") as f:
            for l in f:
                d = json.loads(l)
                retried[d["work_id"]] = len(d["attempts"]) > 1   # last record per abstract wins
        return retried, path
    if os.path.exists(index_path):
        old = pd.read_csv(index_path)
        return dict(zip(old.work_id.astype(str), old.had_retry.astype(bool))), index_path
    sys.exit(f"need {path} (local only) or an existing {index_path} for had_retry")


def mb(path):
    return os.path.getsize(path) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg, acfg = load_config(args.config)
    paths = export_paths(acfg)
    os.makedirs(acfg["exports_dir"], exist_ok=True)

    outputs = read_outputs(acfg["run_dir"])
    retried, retry_source = read_retries(acfg["run_dir"], paths["index"])
    corpus = load_corpus(cfg["corpus"]["path"], cfg["corpus"]["columns"])
    for col in ("period", "strand"):
        if col not in corpus.columns:
            corpus[col] = None
    missing = [o["work_id"] for o in outputs if o["work_id"] not in corpus.index]
    if missing:
        sys.exit(f"{len(missing)} processed abstracts are not in the corpus, e.g. {missing[:3]}")

    docs = corpus.loc[[o["work_id"] for o in outputs]].to_dict("records")
    index, rows = [], []
    for o, doc in zip(outputs, docs):
        chars = len(doc["abstract"])
        status = LONG_TEXT if chars > acfg["long_text_chars"] else RETAINED
        meta = {"work_id": o["work_id"], "year": doc["year"], "period": doc["period"], "strand": doc["strand"]}
        index.append({**meta, "title": doc.get("title"), "abstract_chars": chars, "status": status,
                      "n_components": len(o["components"]), "had_retry": bool(retried.get(o["work_id"], False)),
                      "n_flags": len(o["flags"])})
        rows += [{**meta, "type": c["type"], "label": c["label"], "evidence": c["evidence"],
                  "confidence": c["confidence"], "status": status} for c in o["components"]]
    index = pd.DataFrame(index).sort_values("work_id")
    comps = pd.DataFrame(rows, columns=["work_id", "year", "period", "strand", "type", "label", "evidence",
                                        "confidence", "status"])
    comps = comps.iloc[comps.work_id.argsort(kind="stable")]   # by work_id, model order kept within
    for df in (index, comps):
        df["year"] = df["year"].astype("Int64")

    index.to_csv(paths["index"], index=False)
    comps.to_parquet(paths["components"], index=False, compression="zstd")
    sizes = {paths["components"]: mb(paths["components"]), paths["index"]: mb(paths["index"])}
    if sizes[paths["components"]] >= MAX_MB:
        os.remove(paths["components"])
        comps.to_csv(paths["components_csv"], index=False, compression="gzip")
        sizes = {paths["components_csv"]: mb(paths["components_csv"]), paths["index"]: mb(paths["index"])}
    elif os.path.exists(paths["components_csv"]):
        os.remove(paths["components_csv"])

    # long-text report: overall, and per year where the excluded share of abstracts is above 5%
    long_ix = index[index.status == LONG_TEXT]
    n_long_comps = int((comps.status == LONG_TEXT).sum())
    by_year = index.groupby("year").agg(abstracts=("work_id", "size"),
                                        excluded=("status", lambda s: int((s == LONG_TEXT).sum())))
    cy = comps.groupby("year").agg(components=("work_id", "size"),
                                   comps_excluded=("status", lambda s: int((s == LONG_TEXT).sum())))
    by_year = by_year.join(cy).fillna(0).astype(int)
    by_year["share"] = by_year.excluded / by_year.abstracts
    over = by_year[by_year.share > 0.05]
    md = ["# Long-text rule: what it excludes", "",
          f"Abstracts over {acfg['long_text_chars']:,} characters get `status = {LONG_TEXT}` (probably body text, "
          "not an abstract). They stay in every export with that status; Figure 1 uses "
          f"`status = {RETAINED}` only.", "",
          "| | processed | excluded | excluded % |", "|---|---|---|---|",
          f"| abstracts | {len(index):,} | {len(long_ix):,} | {100 * len(long_ix) / len(index):.1f} |",
          f"| components | {len(comps):,} | {n_long_comps:,} | {100 * n_long_comps / len(comps):.1f} |", "",
          "Years where more than 5% of abstracts are excluded:", ""]
    if len(over):
        md += ["| year | abstracts | excluded | excluded % | components excluded |", "|---|---|---|---|---|"]
        md += [f"| {r.Index} | {r.abstracts:,} | {r.excluded:,} | {100 * r.share:.1f} | {r.comps_excluded:,} |"
               for r in over.itertuples()]
    else:
        md += ["_(none)_"]
    no_year = int(index.year.isna().sum())
    md += ["", f"Processed abstracts with no year in the corpus: {no_year} "
               f"({int(((index.status == LONG_TEXT) & index.year.isna()).sum())} of them excluded); "
               "they are not in the per-year table.", ""]
    open(os.path.join(acfg["exports_dir"], "long_text_report.md"), "w", encoding="utf-8").write("\n".join(md))

    print(f"{len(index):,} abstracts, {len(comps):,} components (had_retry from {retry_source})")
    print(f"excluded_long_text: {len(long_ix):,} abstracts, {n_long_comps:,} components")
    for p, s in sizes.items():
        print(f"{p}: {s:.1f} MB")
    if max(sizes.values()) >= MAX_MB:
        sys.exit(f"an export is still over {MAX_MB} MB (see sizes above)")


if __name__ == "__main__":
    main()
