"""Full-corpus run in legs: python -m src.run --full --cap <USD>

Every eligible abstract (at least full_run.min_chars characters) is processed once, in an order
shuffled once with full_run.seed, so any partial run is a random sample. All legs share one run
folder (full_run.dir). Each abstract's output is appended as it finishes, and a restart skips
abstracts that already have one, so the same command starts or resumes the run.

A leg stops cleanly (in-flight requests finish and are saved) when its spend reaches --cap (budget
is held back for requests in flight, at the dearest abstract seen this leg), when
the API reports insufficient balance or a bad key, or when every abstract is done. The run starts
only off-peak, and pauses itself when a peak window begins and resumes when it ends.

Run folder:
  manifest.json   seed, order hash, prompt/codebook/model at creation; a resume must match it
  order.txt       the shuffled ids
  outputs.jsonl   one validated output per finished abstract
  details.jsonl   one record per processed abstract (also for API errors), with cost
  errors.jsonl    abstracts that ended in a per-abstract API error; retried by the next leg
  legs.json       one entry per leg: times, workers, stop reason, spend, pauses, balance
  legs.md         the report printed at the end of each leg
  run_log.json    cumulative run log, rewritten at the end of each leg
"""
import datetime as dt
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import requests

from src.extract import FatalApiError, extract, is_peak


def now():
    return dt.datetime.now(dt.timezone.utc)


def peak_ends(t, pricing):
    """First whole hour at or after t that is off-peak."""
    t = t.replace(minute=0, second=0, microsecond=0)
    while is_peak(t, pricing):
        t += dt.timedelta(hours=1)
    return t


def eligible_ids(corpus, min_chars):
    ab = corpus["abstract"]
    ok = ab.apply(lambda x: isinstance(x, str) and len(x) >= min_chars)
    return sorted(corpus.index[ok])


def shuffled(ids, seed):
    return [ids[k] for k in np.random.default_rng(seed).permutation(len(ids))]


def read_jsonl(path, strip=()):
    """Read a JSONL file, cutting off a trailing partial line left by a crash."""
    if not os.path.exists(path):
        return []
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)
            data = data[:data.rfind(b"\n") + 1]
    rows = []
    for line in data.decode("utf-8").splitlines():
        r = json.loads(line)
        for a in r.get("attempts", []) + r.get("resolutions", []) if strip else []:
            for k in strip:
                a.pop(k, None)
        rows.append(r)
    return rows


def balance(model_cfg):
    """Account balance from the provider's balance endpoint, or None if it can't be read."""
    try:
        r = requests.get(model_cfg["base_url"].rstrip("/") + "/user/balance", timeout=30,
                         headers={"Authorization": f'Bearer {os.environ[model_cfg["api_key_env"]]}'})
        r.raise_for_status()
        return r.json().get("balance_infos")
    except (requests.RequestException, ValueError, KeyError):
        return None


def append(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def run_full(args, cfg, corpus, system_prompt, codebook, hashes, summarise):
    fr = cfg["full_run"]
    out_dir = args.run_dir or fr["dir"]
    pricing = cfg["pricing"]

    if is_peak(now(), pricing):
        sys.exit(f"Peak hours now ({now():%a %H:%M} UTC). The run starts only off-peak; "
                 f"next off-peak from {peak_ends(now(), pricing):%a %H:%M} UTC.")

    order = shuffled(eligible_ids(corpus, fr["min_chars"]), fr["seed"])
    if args.limit:
        order = order[:args.limit]
    manifest = {"seed": fr["seed"], "min_chars": fr["min_chars"], "limit": args.limit, "n": len(order),
                "order_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
                "model_id": cfg["model"]["id"], **hashes}
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "manifest.json")
    if os.path.exists(mpath):
        old = json.load(open(mpath, encoding="utf-8"))
        diff = [k for k in manifest if old.get(k) != manifest[k]]
        if diff:
            sys.exit(f"{out_dir} was started with different {diff}; refusing to resume. "
                     f"Was: {[old.get(k) for k in diff]}, now: {[manifest[k] for k in diff]}.")
    else:
        manifest["created_utc"] = now().isoformat(timespec="seconds")
        json.dump(manifest, open(mpath, "w", encoding="utf-8"), indent=2)
        open(os.path.join(out_dir, "order.txt"), "w", encoding="utf-8").write("\n".join(order) + "\n")

    paths = {k: os.path.join(out_dir, f"{k}.jsonl") for k in ("outputs", "details", "errors")}
    done = {o["work_id"] for o in read_jsonl(paths["outputs"])}
    lpath = os.path.join(out_dir, "legs.json")
    legs = json.load(open(lpath, encoding="utf-8")) if os.path.exists(lpath) else []
    spent_before = (sum(d["cost_usd"] for d in read_jsonl(paths["details"], strip=("raw",)))
                    + sum(l["unsaved_spend_usd"] for l in legs))
    todo = [i for i in order if i not in done]
    if not todo:
        sys.exit(f"{out_dir}: all {len(order)} abstracts are done.")

    w = fr["workers"]
    leg = {"leg": len(legs) + 1, "start_utc": now().isoformat(timespec="seconds"), "cap_usd": args.cap,
           "balance_start": balance(cfg["model"]), "workers": [[0, w["start"]]], "pauses": [],
           "http_429": 0, "api_errors": 0, "done": 0, "spend_usd": 0.0, "unsaved_spend_usd": 0.0}
    for l in legs:
        if l.get("stop_reason") is None:
            l["stop_reason"] = "killed (no clean stop); its saved abstracts are counted, spend on its in-flight requests is not"
    legs.append(leg)
    json.dump(legs, open(lpath, "w", encoding="utf-8"), indent=2)  # so a crashed leg is still on record
    print(f"Leg {leg['leg']} of {out_dir}: {len(done)} of {len(order)} done, {len(todo)} to go, "
          f"cap ${args.cap:.4f}, spent so far ${spent_before:.4f}, {w['start']} workers.", flush=True)

    t0 = time.monotonic()
    workers, stop, max_cost = w["start"], None, 0.002
    nxt = 0  # index into todo of the next abstract to submit
    inflight = {}
    files = {k: open(p, "a", encoding="utf-8") for k, p in paths.items()}
    pool = ThreadPoolExecutor(max_workers=max(w["start"], w["raise_to"]))

    def one(i):
        return extract(i, corpus.at[i, "abstract"], system_prompt, codebook, cfg)

    def collect(timeout):
        nonlocal stop, max_cost
        finished, _ = wait(inflight, timeout=timeout, return_when=FIRST_COMPLETED)
        for fut in finished:
            i = inflight.pop(fut)
            try:
                out, det = fut.result()
            except FatalApiError as e:
                leg["unsaved_spend_usd"] += getattr(e, "cost_usd", 0.0)
                stop = stop or f"API: {e}"
                continue
            except Exception as e:  # anything unexpected: stop cleanly rather than lose track
                stop = stop or f"error on {i}: {type(e).__name__}: {e}"
                continue
            det["leg"] = leg["leg"]
            max_cost = max(max_cost, det["cost_usd"])
            leg["spend_usd"] += det["cost_usd"]
            leg["http_429"] += det["http_429"]
            append(files["details"], det)
            if det["api_error"]:
                leg["api_errors"] += 1
                append(files["errors"], {"work_id": i, "leg": leg["leg"], "error": det["api_error"],
                                         "utc": now().isoformat(timespec="seconds")})
            else:
                append(files["outputs"], out)
                done.add(i)
                leg["done"] += 1

    try:
        while True:
            if workers < w["raise_to"] and leg["http_429"] == 0 and time.monotonic() - t0 >= w["after_minutes"] * 60:
                workers = w["raise_to"]
                leg["workers"].append([round((time.monotonic() - t0) / 60, 1), workers])
                print(f"No rate-limit errors in {w['after_minutes']} min: raising to {workers} workers.", flush=True)
            # Hold back budget for requests in flight at the dearest abstract seen this leg (at least
            # $0.002), so the cap isn't overshot by a batch. Cheap abstracts finish first, so a mean
            # would underestimate what is still in flight.
            est = max_cost
            if stop is None and nxt < len(todo) and leg["spend_usd"] + (len(inflight) + 1) * est > args.cap:
                if inflight:
                    collect(5)
                    continue
                stop = f"cap reached (${leg['spend_usd']:.4f} spent; cap ${args.cap:.4f}, next abstract ~${est:.4f})"
            if stop is None and is_peak(now(), pricing):
                if inflight:
                    collect(5)
                    continue
                until = peak_ends(now(), pricing)
                pause = {"from_utc": now().isoformat(timespec="seconds"), "until_utc": until.isoformat()}
                leg["pauses"].append(pause)
                print(f"Peak window began: paused at {now():%H:%M} UTC until {until:%H:%M} UTC.", flush=True)
                while is_peak(now(), pricing):
                    time.sleep(30)
                pause["resumed_utc"] = now().isoformat(timespec="seconds")
                print(f"Off-peak again: resumed at {now():%H:%M} UTC.", flush=True)
            while stop is None and len(inflight) < workers and nxt < len(todo):
                inflight[pool.submit(one, todo[nxt])] = todo[nxt]
                nxt += 1
            if not inflight:
                stop = stop or "finished (every abstract in the queue processed)"
                break
            collect(5)
            if stop is not None and not inflight:
                break
    except KeyboardInterrupt:
        stop = "interrupted"
        print("Interrupted: finishing in-flight requests…", flush=True)
        while inflight:
            collect(5)
    finally:
        pool.shutdown(wait=True)
        for f in files.values():
            f.close()
        leg["end_utc"] = now().isoformat(timespec="seconds")
        leg["stop_reason"] = stop
        leg["balance_end"] = balance(cfg["model"])
        leg["spend_usd"] = round(leg["spend_usd"], 6)
        json.dump(legs, open(lpath, "w", encoding="utf-8"), indent=2)
        report = leg_report(out_dir, corpus, order, done, legs, spent_before, fr.get("report_by"))
        open(os.path.join(out_dir, "legs.md"), "a", encoding="utf-8").write(report + "\n")
        write_log(out_dir, paths, legs, manifest, cfg, summarise, len(order))
        print(report, flush=True)


def leg_report(out_dir, corpus, order, done, legs, spent_before, by):
    leg = legs[-1]
    total = spent_before + leg["spend_usd"] + leg["unsaved_spend_usd"]
    n_done, n_all = len(done), len(order)
    md = [f"## Leg {leg['leg']}: {leg['start_utc']} to {leg['end_utc']}", "",
          f"- Stopped: {leg['stop_reason']}",
          f"- Done: {n_done:,} of {n_all:,} ({100 * n_done / n_all:.1f}%); remaining {n_all - n_done:,}; "
          f"done this leg {leg['done']:,}; API errors this leg {leg['api_errors']} (retried next leg)",
          f"- Spend: this leg ${leg['spend_usd'] + leg['unsaved_spend_usd']:.4f} (cap ${leg['cap_usd']:.4f}), "
          f"total ${total:.4f}"
          + (f", of which ${leg['unsaved_spend_usd']:.4f} on abstracts cut off by an API stop"
             if leg["unsaved_spend_usd"] else ""),
          f"- Workers: {', '.join(f'{n} from minute {m}' for m, n in leg['workers'])}; HTTP 429 this leg: {leg['http_429']}",
          f"- Paused for peak: {len(leg['pauses'])} time(s)"
          + "".join(f"; {p['from_utc']} to {p.get('resumed_utc', '(stopped)')}" for p in leg["pauses"])]
    if n_done:
        md.append(f"- Estimated cost to finish: ${total / n_done * (n_all - n_done):.2f} "
                  f"(mean ${total / n_done:.6f} per done abstract × {n_all - n_done:,} remaining, off-peak rates)")
    md.append(f"- Balance reported by the API: start {fmt_bal(leg['balance_start'])}, end {fmt_bal(leg['balance_end'])}")
    if by and by in corpus.columns:
        col = corpus.loc[order, by].fillna("(none)")
        tot, dn = col.value_counts(), col[col.index.isin(done)].value_counts()
        md += ["", f"| {by} | done | eligible | share done |", "|---|---|---|---|"]
        md += [f"| {k} | {dn.get(k, 0):,} | {n:,} | {100 * dn.get(k, 0) / n:.1f}% |" for k, n in tot.sort_index().items()]
    return "\n".join(md) + "\n"


def fmt_bal(b):
    if not b:
        return "unavailable"
    return ", ".join(f"{x.get('total_balance')} {x.get('currency')}" for x in b)


def write_log(out_dir, paths, legs, manifest, cfg, summarise, n_order):
    outputs = read_jsonl(paths["outputs"])
    details = read_jsonl(paths["details"], strip=("raw",))
    latest = {d["work_id"]: d for d in details}
    done_ids = {o["work_id"] for o in outputs}
    s = summarise([latest[i] for i in done_ids], outputs)
    usage = Counter()
    for d in details:
        usage.update({k: v for k, v in d["usage"].items() if isinstance(v, (int, float))})
    errors = read_jsonl(paths["errors"])
    log = {
        "run_dir": out_dir, **manifest,
        "model": {"provider": cfg["model"]["provider"], "id": cfg["model"]["id"],
                  "settings": cfg["model"].get("params", {})},
        "n_abstracts": n_order, "n_done": len(done_ids),
        "n_with_error_not_done": len({e["work_id"] for e in errors} - done_ids),
        **s,
        "summary_note": "counts are over done abstracts; tokens and cost are over every request "
                        "made in all legs, including abstracts that ended in an API error",
        "tokens": {"input": usage.get("prompt_tokens", 0), "input_cache_hit": usage.get("prompt_cache_hit_tokens", 0),
                   "input_cache_miss": usage.get("prompt_cache_miss_tokens", 0), "output": usage.get("completion_tokens", 0)},
        "cost_usd": round(sum(d["cost_usd"] for d in details) + sum(l["unsaved_spend_usd"] for l in legs), 6),
        "api_errors": len(errors),
        "http_429": sum(l["http_429"] for l in legs),
        "legs": legs,
        "pricing": {k: cfg["pricing"][k] for k in ("source", "checked", "peak", "off_peak", "peak_hours_utc")},
    }
    log.pop("api_error_messages", None)
    json.dump(log, open(os.path.join(out_dir, "run_log.json"), "w", encoding="utf-8"), indent=2)
