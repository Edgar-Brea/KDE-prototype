"""One abstract in, validated JSON out.

The system prompt is the prompt file (codebook + machine instructions). The user message
carries the output schema and the abstract only (never the title). Overlaps left after the
retry go to one resolution call per pair, with the codebook alone as system prompt.
"""
import datetime as dt
import json
import os
import time

import requests

from src.validate import CONFIDENCE, TYPES, check_schema, overlap_pairs, resolve_overlaps, validate

USER_TEMPLATE = (
    "Return a JSON object with this schema:\n"
    '{"components": [{"type": "' + "|".join(TYPES) + '", '
    '"label": "a few words", "evidence": "shortest exact words from the abstract", '
    '"confidence": "' + "|".join(CONFIDENCE) + '"}]}\n\n'
    "Abstract:\n{abstract}"
)

RETRY_TEMPLATE = (
    "Your answer had these problems:\n{problems}\n\n"
    "Return the complete corrected JSON for the same abstract, using the same schema."
)

RESOLVE_TEMPLATE = (
    "Two components coded from this abstract have overlapping evidence but different types. "
    "Each phrase gets one type only. Using the codebook, decide which single type fits the evidence.\n\n"
    "Abstract:\n{abstract}\n\n"
    'A: type {a_type}, label "{a_label}", evidence "{a_evidence}"\n'
    'B: type {b_type}, label "{b_label}", evidence "{b_evidence}"\n\n'
    'Return a JSON object: {"keep": "A"} or {"keep": "B"}.'
)


def load_env(path=".env"):
    """Read KEY=VALUE lines into os.environ without overriding variables already set."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def is_peak(t, pricing):
    if t.weekday() >= 5:
        return False
    return any(a <= t.hour < b for a, b in pricing["peak_hours_utc"])


def call_cost(usage, t, pricing):
    rates = pricing["peak" if is_peak(t, pricing) else "off_peak"]
    hit = usage.get("prompt_cache_hit_tokens", 0)
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
    out = usage.get("completion_tokens", 0)
    return (hit * rates["input_cache_hit"] + miss * rates["input_cache_miss"] + out * rates["output"]) / 1e6


class ApiError(Exception):
    pass


class FatalApiError(Exception):
    """An error that will hit every request (no balance, bad key). Not an ApiError, so it is never
    caught per abstract: it propagates to the runner, which stops."""


FATAL_STATUS = {401: "authentication failed", 402: "insufficient balance"}


def chat(model_cfg, messages):
    """POST one chat completion. Retries transient failures.

    Returns (response json, n transient retries, n of them that were HTTP 429).
    """
    key = os.environ.get(model_cfg["api_key_env"])
    if not key:
        raise ApiError(f'environment variable {model_cfg["api_key_env"]} is not set')
    url = model_cfg["base_url"].rstrip("/") + "/chat/completions"
    body = {"model": model_cfg["id"], "messages": messages, **model_cfg.get("params", {})}
    retries = rate_limited = 0
    for attempt in range(model_cfg.get("max_http_retries", 4) + 1):
        try:
            r = requests.post(url, json=body, timeout=model_cfg.get("timeout_s", 120),
                              headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 200:
                return r.json(), retries, rate_limited
            err = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in FATAL_STATUS:
                raise FatalApiError(f"{FATAL_STATUS[r.status_code]} ({err})")
            transient = r.status_code == 429 or r.status_code >= 500
            rate_limited += r.status_code == 429
        except requests.RequestException as e:
            err, transient = f"{type(e).__name__}: {e}", True
        if not transient or attempt == model_cfg.get("max_http_retries", 4):
            raise ApiError(err)
        retries += 1
        time.sleep(2 ** attempt * 2)


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace("{" + k + "}", v)
    return template


def extract(work_id, abstract, system_prompt, codebook, cfg):
    """Run one abstract. Returns (output, detail).

    output — {"work_id", "components", "flags"}: only components that passed validation,
             and one flag per overlap conflict settled by a resolution call.
    detail — attempts, resolutions, failures, drops, usage and cost, for the run log.
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_TEMPLATE.replace("{abstract}", abstract)}]
    detail = {"work_id": work_id, "attempts": [], "resolutions": [], "overlap": {}, "dropped": [],
              "usage": {}, "cost_usd": 0.0, "api_error": None, "http_retries": 0, "http_429": 0}
    usage_tot = {}
    final_results = []

    def call(msgs):
        """One API call. Returns (raw content, record) and adds usage and cost to the totals."""
        t = dt.datetime.now(dt.timezone.utc)
        try:
            resp, n_retry, n_429 = chat(cfg["model"], msgs)
        except FatalApiError as e:
            e.cost_usd = detail["cost_usd"]  # spent on this abstract before the error, for the runner's total
            raise
        detail["http_retries"] += n_retry
        detail["http_429"] += n_429
        usage = resp.get("usage", {}) or {}
        for k, v in usage.items():
            if isinstance(v, (int, float)):
                usage_tot[k] = usage_tot.get(k, 0) + v
        detail["cost_usd"] += call_cost(usage, t, cfg["pricing"])
        raw = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return raw, {"model_returned": resp.get("model"), "system_fingerprint": resp.get("system_fingerprint"),
                     "peak": is_peak(t, cfg["pricing"]), "usage": usage, "raw": raw}

    def decide(a, b):
        msgs = [{"role": "system", "content": codebook},
                {"role": "user", "content": fill(
                    RESOLVE_TEMPLATE, abstract=abstract,
                    a_type=a["type"], a_label=a["label"], a_evidence=a["evidence"],
                    b_type=b["type"], b_label=b["label"], b_evidence=b["evidence"])}]
        rec = {"pair": [a, b], "keep": None, "error": None}
        detail["resolutions"].append(rec)
        try:
            raw, info = call(msgs)
            rec.update(info)
            keep = json.loads(raw).get("keep")
            if keep not in ("A", "B"):
                rec["error"] = f'answer {raw.strip()[:200]!r} does not choose "A" or "B"'
        except ApiError as e:
            rec["error"] = f"API error: {e}"
        except (json.JSONDecodeError, AttributeError) as e:
            rec["error"] = f"response is not a JSON object ({e})"
        if rec["error"]:
            return None, rec["error"]
        rec["keep"] = keep
        return keep, None

    for attempt in (1, 2):
        try:
            raw, info = call(messages)
        except ApiError as e:
            detail["api_error"] = str(e)
            break
        rec = {"attempt": attempt, **info, "failures": []}
        try:
            components, err = check_schema(json.loads(raw))
        except json.JSONDecodeError as e:
            components, err = None, f"response is not valid JSON ({e})"
        if components is None:
            rec["failures"].append({"index": None, "kind": "json", "reason": err})
            results = []
        else:
            results, failures = validate(components, abstract)
            rec["failures"] = failures
        detail["attempts"].append(rec)
        final_results = results

        if not rec["failures"] or attempt == 2:
            break
        problems = list(dict.fromkeys("- " + f["reason"] for f in rec["failures"]))
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": RETRY_TEMPLATE.replace("{problems}", "\n".join(problems))}]

    flags = []
    pairs = overlap_pairs(detail["attempts"][-1]["failures"]) if detail["attempts"] else []
    detail["overlap"] = {"first_try": len(overlap_pairs(detail["attempts"][0]["failures"])) if detail["attempts"] else 0,
                         "after_retry": len(pairs), "resolved": 0, "dropped": 0, "moot": 0}
    if pairs:
        flags, counts = resolve_overlaps(final_results, pairs, decide)
        detail["overlap"].update(counts)

    detail["usage"] = usage_tot
    kept, detail["spans"] = [], []
    for r in final_results:
        if r["ok"]:
            c = r["component"]
            kept.append({"type": c["type"], "label": c["label"], "evidence": c["evidence"],
                         "confidence": c["confidence"]})
            detail["spans"].append(list(r["span"]))
        else:
            detail["dropped"].append({"component": r["component"], "reasons": r["reasons"]})
    if detail["attempts"] and detail["attempts"][-1]["failures"] and not final_results:
        # Unparseable final answer: nothing to keep, record why.
        detail["dropped"].append({"component": None,
                                  "reasons": [f["reason"] for f in detail["attempts"][-1]["failures"]]})
    return {"work_id": work_id, "components": kept, "flags": flags}, detail
