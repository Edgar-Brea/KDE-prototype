"""Batch runner: extract components for a list of IDs and write a run folder.

  python -m src.run --ids example/ids.txt --name example
  python -m src.run --full --cap 5.00        # full corpus: starts or resumes, see src/full.py

Writes runs/<run_id>/: outputs.jsonl (one {"work_id", "components", "flags"} per abstract),
details.jsonl (attempts, resolution calls, failures, drops, usage per abstract) and run_log.json.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yaml

from src.extract import RESOLVE_TEMPLATE, RETRY_TEMPLATE, USER_TEMPLATE, extract, load_env
from src.validate import CONFIDENCE, TYPES


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_frozen(files, frozen_path):
    """Exit unless every file is the one recorded in FROZEN.

    files maps a role to a path, e.g. {"prompt": ..., "codebook": ...}. FROZEN has one line per
    role, `<role> <filename> <sha256>`, and must list exactly these roles.
    """
    if not os.path.exists(frozen_path):
        sys.exit(f"{frozen_path} not found: nothing is frozen, so runs are refused.")
    frozen = {}
    for line in open(frozen_path, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            role, name, digest = line.split()
            frozen[role] = (name, digest)
    if set(frozen) != set(files):
        sys.exit(f"{frozen_path} lists {sorted(frozen)}, expected {sorted(files)}. Run refused.")
    for role, path in files.items():
        name, digest = frozen[role]
        if os.path.basename(path) != name or sha256(path) != digest:
            sys.exit(f"{role.capitalize()} {path} (sha256 {sha256(path)}) does not match the frozen "
                     f"{role} in {frozen_path} ({name}, sha256 {digest}). Run refused.")
    return frozen


def load_corpus(path, columns):
    """Load a .parquet/.csv/.jsonl corpus and rename its columns to id, title, abstract, year."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={v: k for k, v in columns.items()})
    missing = {"id", "abstract"} - set(df.columns)
    if missing:
        sys.exit(f"corpus {path} has no column for {sorted(missing)}; check corpus.columns in the config")
    df["id"] = df["id"].astype(str)
    return df.set_index("id")


def read_ids(path):
    ids = [l.strip() for l in open(path, encoding="utf-8")]
    return [i for i in ids if i and not i.startswith("#")]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def summarise(details, outputs):
    first = Counter()
    final = Counter()
    retried = 0
    for d in details:
        if d["attempts"]:
            # Overlaps are counted once per pair, everything else once per component.
            first.update(f["kind"] for f in d["attempts"][0]["failures"] if f["kind"] != "overlap")
            final.update(f["kind"] for f in d["attempts"][-1]["failures"] if f["kind"] != "overlap")
            retried += len(d["attempts"]) > 1
    overlap = Counter()
    for d in details:
        overlap.update(d["overlap"])
    first["overlap"], final["overlap"] = overlap["first_try"], overlap["after_retry"]
    usage = Counter()
    for d in details:
        usage.update(d["usage"])
    per_type = Counter(c["type"] for o in outputs for c in o["components"])
    per_conf = Counter(c["confidence"] for o in outputs for c in o["components"])
    per_type_conf = Counter((c["type"], c["confidence"]) for o in outputs for c in o["components"])
    kinds = ("span", "overlap", "type", "confidence", "field", "json")
    return {
        "n_with_components": sum(bool(o["components"]) for o in outputs),
        "components_total": sum(per_type.values()),
        "components_per_type": {t: per_type.get(t, 0) for t in TYPES},
        "components_per_confidence": {c: per_conf.get(c, 0) for c in CONFIDENCE},
        "components_per_type_confidence": {t: {c: per_type_conf.get((t, c), 0) for c in CONFIDENCE} for t in TYPES},
        "abstracts_retried": retried,
        "failures_first_try": {k: first.get(k, 0) for k in kinds},
        "failures_after_retry": {k: final.get(k, 0) for k in kinds},
        "failure_count_unit": "components, except overlap failures, which are counted in pairs",
        "span_failures": {"first_try": first.get("span", 0), "after_retry": final.get("span", 0)},
        "overlap_failures": {k: overlap.get(k, 0) for k in ("first_try", "after_retry", "resolved", "dropped", "moot")},
        "overlap_failures_note": "pairs. after_retry = resolved + dropped + moot; moot = pair whose other "
                                 "member was already dropped by an earlier resolution; dropped = resolution "
                                 "call failed, both components dropped",
        "resolution_calls": sum(len(d["resolutions"]) for d in details),
        "resolution_call_errors": {d["work_id"]: [r["error"] for r in d["resolutions"] if r["error"]]
                                   for d in details if any(r["error"] for r in d["resolutions"])},
        "components_dropped": sum(1 for d in details for x in d["dropped"] if x["component"] is not None),
        "api_errors": sum(1 for d in details if d["api_error"]),
        "api_error_messages": {d["work_id"]: d["api_error"] for d in details if d["api_error"]},
        "http_transient_retries": sum(d["http_retries"] for d in details),
        "tokens": {
            "input": usage.get("prompt_tokens", 0),
            "input_cache_hit": usage.get("prompt_cache_hit_tokens", 0),
            "input_cache_miss": usage.get("prompt_cache_miss_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        },
        "cost_usd": round(sum(d["cost_usd"] for d in details), 6),
        "calls_at_peak": sum(a["peak"] for d in details for a in d["attempts"] + d["resolutions"] if "peak" in a),
        "calls_off_peak": sum(not a["peak"] for d in details for a in d["attempts"] + d["resolutions"] if "peak" in a),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ids", help="text file, one id per line")
    mode.add_argument("--full", action="store_true", help="start or resume the full-corpus run (needs --cap)")
    ap.add_argument("--cap", type=float, help="--full: stop this leg when its spend reaches this many USD")
    ap.add_argument("--limit", type=int, help="--full: only the first N abstracts of the shuffled order (tests)")
    ap.add_argument("--run-dir", help="--full: run folder, overriding full_run.dir in the config")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--corpus", help="override corpus.path from the config")
    ap.add_argument("--prompt", help="override prompt from the config")
    ap.add_argument("--codebook", help="override codebook (used by overlap resolution calls) from the config")
    ap.add_argument("--name", help="suffix for the run folder name, e.g. example")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.full and args.cap is None:
        ap.error("--full needs --cap <USD>")

    load_env()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    prompt_path = args.prompt or cfg["prompt"]
    corpus_path = args.corpus or cfg["corpus"]["path"]
    codebook_path = args.codebook or cfg["codebook"]
    frozen_path = cfg.get("frozen", os.path.join(os.path.dirname(prompt_path), "FROZEN"))
    check_frozen({"prompt": prompt_path, "codebook": codebook_path}, frozen_path)
    system_prompt = open(prompt_path, encoding="utf-8").read()
    codebook = open(codebook_path, encoding="utf-8").read()
    if not os.environ.get(cfg["model"]["api_key_env"]):
        sys.exit(f'Set {cfg["model"]["api_key_env"]} (environment or .env) before running.')

    corpus = load_corpus(corpus_path, cfg["corpus"]["columns"])
    if args.full:
        from src.full import run_full
        hashes = {"prompt_file": prompt_path, "prompt_sha256": sha256(prompt_path),
                  "codebook_file": codebook_path, "codebook_sha256": sha256(codebook_path)}
        return run_full(args, cfg, corpus, system_prompt, codebook, hashes, summarise)
    ids = read_ids(args.ids)
    unknown = [i for i in ids if i not in corpus.index]
    if unknown:
        sys.exit(f"{len(unknown)} ids not in corpus, e.g. {unknown[:5]}")
    no_abs = [i for i in ids if not isinstance(corpus.at[i, "abstract"], str) or not corpus.at[i, "abstract"].strip()]
    if no_abs:
        sys.exit(f"{len(no_abs)} ids have no abstract, e.g. {no_abs[:5]}")

    started = dt.datetime.now(dt.timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + (f"_{args.name}" if args.name else "")
    out_dir = os.path.join("runs", run_id)
    os.makedirs(out_dir)

    def one(i):
        out, det = extract(i, corpus.at[i, "abstract"], system_prompt, codebook, cfg)
        print(f"  {i}: {len(out['components'])} components"
              + (f", {len(det['dropped'])} dropped" if det["dropped"] else "")
              + (f", {len(out['flags'])} overlap flags" if out["flags"] else "")
              + (f", API ERROR {det['api_error']}" if det["api_error"] else ""), flush=True)
        return out, det

    print(f"Run {run_id}: {len(ids)} abstracts, model {cfg['model']['id']}, prompt {prompt_path}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, ids))
    outputs = [r[0] for r in results]
    details = [r[1] for r in results]

    with open(os.path.join(out_dir, "outputs.jsonl"), "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "details.jsonl"), "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    finished = dt.datetime.now(dt.timezone.utc)
    returned = sorted({a["model_returned"] for d in details for a in d["attempts"] if a["model_returned"]})
    log = {
        "run_id": run_id,
        "timestamp_start_utc": started.isoformat(timespec="seconds"),
        "timestamp_end_utc": finished.isoformat(timespec="seconds"),
        "duration_s": round((finished - started).total_seconds(), 1),
        "git_commit": git_commit(),
        "model": {"provider": cfg["model"]["provider"], "base_url": cfg["model"]["base_url"],
                  "id": cfg["model"]["id"], "returned_by_api": returned,
                  "settings": cfg["model"].get("params", {}),
                  "settings_note": "lowest reasoning effort = thinking disabled; temperature 0 "
                                   "(honoured in non-thinking mode); top_p fixed at 1.0 by the API"},
        "prompt_file": prompt_path,
        "prompt_sha256": sha256(prompt_path),
        "frozen": {"file": frozen_path, "prompt_matches": True, "codebook_matches": True},
        "user_message_sha256": hashlib.sha256((USER_TEMPLATE + RETRY_TEMPLATE).encode()).hexdigest(),
        "codebook_file": codebook_path,
        "codebook_sha256": sha256(codebook_path),
        "resolution_message_sha256": hashlib.sha256(RESOLVE_TEMPLATE.encode()).hexdigest(),
        "corpus": corpus_path,
        "ids_file": args.ids,
        "n_abstracts": len(ids),
        **summarise(details, outputs),
        "pricing": {k: cfg["pricing"][k] for k in ("source", "checked", "peak", "off_peak", "peak_hours_utc")},
    }
    json.dump(log, open(os.path.join(out_dir, "run_log.json"), "w", encoding="utf-8"), indent=2)
    print(f"Done: {out_dir}  components {log['components_total']}  dropped {log['components_dropped']}  "
          f"API errors {log['api_errors']}  cost ${log['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
