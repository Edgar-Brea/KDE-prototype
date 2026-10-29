"""Checks on model output: schema (type and confidence values), verbatim evidence, no cross-type span overlap.

Evidence matching collapses whitespace and normalises quotes and dashes, nothing else.
Spans are never repaired: a component either matches or is reported as a failure.
Cross-type overlaps left after the retry are settled by resolve_overlaps.
"""

TYPES = ("phenomenon", "principle", "means", "purpose")
CONFIDENCE = ("high", "medium", "low")

_QUOTES = {c: "'" for c in "‘’‚‛′"}
_QUOTES.update({c: '"' for c in "“”„‟″«»"})
_DASHES = {c: "-" for c in "‐‑‒–—―−﹘﹣－"}
_CHARMAP = {**_QUOTES, **_DASHES}


def normalise(text, charmap=_CHARMAP):
    """Return (normalised text, index map). index[i] = position in `text` of normalised char i."""
    out, index = [], []
    pending_space = None
    for i, ch in enumerate(text):
        if ch.isspace():
            if pending_space is None:
                pending_space = i
            continue
        if pending_space is not None and out:
            out.append(" ")
            index.append(pending_space)
        pending_space = None
        out.append(charmap.get(ch, ch))
        index.append(i)
    return "".join(out), index


def find_spans(evidence, abstract_norm, index, charmap=_CHARMAP):
    """All (start, end) character spans in the original abstract where `evidence` occurs."""
    ev, _ = normalise(evidence, charmap)
    if not ev:
        return []
    spans, start = [], abstract_norm.find(ev)
    while start != -1:
        end = start + len(ev) - 1
        spans.append((index[start], index[end] + 1))
        start = abstract_norm.find(ev, start + 1)
    return spans


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def check_schema(obj):
    """Return (components, error). components is a list of dicts or None if the shape is wrong."""
    if not isinstance(obj, dict) or not isinstance(obj.get("components"), list):
        return None, 'response must be a JSON object with a "components" list'
    return obj["components"], None


def validate(components, abstract):
    """Validate components against the abstract.

    Returns (results, failures):
      results  — one dict per input component: {"component", "span" or None, "ok", "reasons"}
      failures — list of {"index", "kind", "reason"}; kind is type|confidence|field|span|overlap
    """
    abstract_norm, index = normalise(abstract)
    results, failures = [], []

    def fail(i, kind, reason):
        results[i]["ok"] = False
        results[i]["reasons"].append(reason)
        failures.append({"index": i, "kind": kind, "reason": reason})

    candidates = []
    for i, c in enumerate(components):
        results.append({"component": c, "span": None, "ok": True, "reasons": []})
        candidates.append([])
        if not isinstance(c, dict):
            fail(i, "field", f"component {i + 1} is not an object")
            continue
        if c.get("type") not in TYPES:
            fail(i, "type", f'component {i + 1}: type {c.get("type")!r} is not one of {", ".join(TYPES)}')
        if c.get("confidence") not in CONFIDENCE:
            fail(i, "confidence",
                 f'component {i + 1}: confidence {c.get("confidence")!r} is not one of {", ".join(CONFIDENCE)}')
        for f in ("label", "evidence"):
            if not isinstance(c.get(f), str) or not c.get(f).strip():
                fail(i, "field", f"component {i + 1}: {f} is missing or empty")
        if not results[i]["ok"]:
            continue
        spans = find_spans(c["evidence"], abstract_norm, index)
        if not spans:
            fail(i, "span", f'component {i + 1}: evidence "{c["evidence"]}" is not an exact substring of the abstract')
            continue
        candidates[i] = spans
        results[i]["span"] = spans[0]

    # Evidence that occurs more than once: prefer an occurrence that avoids cross-type overlap.
    placed = [i for i, r in enumerate(results) if r["ok"]]

    def conflicts(i, span):
        t = components[i]["type"]
        return [j for j in placed if j != i and components[j]["type"] != t
                and _overlap(span, results[j]["span"])]

    for i in placed:
        if len(candidates[i]) > 1 and conflicts(i, results[i]["span"]):
            for s in candidates[i]:
                if not conflicts(i, s):
                    results[i]["span"] = s
                    break

    # Any remaining cross-type overlap flags both components: neither is preferred.
    for a in range(len(placed)):
        for b in range(a + 1, len(placed)):
            i, j = placed[a], placed[b]
            if components[i]["type"] != components[j]["type"] and _overlap(results[i]["span"], results[j]["span"]):
                msg = (f'components {i + 1} ({components[i]["type"]}, "{components[i]["evidence"]}") and '
                       f'{j + 1} ({components[j]["type"]}, "{components[j]["evidence"]}") have overlapping '
                       f"evidence but different types")
                fail(i, "overlap", msg)
                fail(j, "overlap", msg)
                failures[-1]["pair"] = failures[-2]["pair"] = [i, j]
    return results, failures


def overlap_pairs(failures):
    """Distinct (i, j) component pairs flagged for cross-type overlap, in order."""
    return list(dict.fromkeys(tuple(f["pair"]) for f in failures if f["kind"] == "overlap"))


def _public(c):
    return {"type": c["type"], "label": c["label"], "evidence": c["evidence"], "confidence": c["confidence"]}


def resolve_overlaps(results, pairs, decide):
    """Settle each overlapping pair by keeping one component and dropping the other.

    decide(a, b) returns ("A" or "B", None) or (None, error message). Pairs are handled in
    order; a pair whose other member was already dropped is moot. If decide fails, both
    components of the pair are dropped. Updates results in place.

    Returns (flags, counts): flags as recorded in the output, counts of resolved/dropped/moot pairs.
    """
    flags, lost = [], set()
    counts = {"resolved": 0, "dropped": 0, "moot": 0}
    for i, j in pairs:
        if i in lost or j in lost:
            counts["moot"] += 1
            continue
        a, b = results[i]["component"], results[j]["component"]
        keep, err = decide(a, b)
        if err:
            lost.update((i, j))
            counts["dropped"] += 1
            for k in (i, j):
                results[k]["reasons"].append(f"resolution call failed, both dropped: {err}")
            flags.append({"decision": "both_dropped", "components": [_public(a), _public(b)], "error": err})
            continue
        win, lose = (i, j) if keep == "A" else (j, i)
        lost.add(lose)
        counts["resolved"] += 1
        results[lose]["reasons"].append(
            f'resolution call kept {results[win]["component"]["type"]} "{results[win]["component"]["label"]}"')
        flags.append({"decision": "resolved", "kept": _public(results[win]["component"]),
                      "dropped": _public(results[lose]["component"])})
    involved = {k for p in pairs for k in p}
    for k in involved - lost:
        results[k]["ok"] = True
        results[k]["reasons"] = []
    return flags, counts
