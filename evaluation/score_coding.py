"""Score the model's components against a human coder's on the same abstracts.

  cd tool && PYTHONPATH=. python ../evaluation/score_coding.py [--config config.yaml]

Reads, all from this repository: the model's output on the test set (`model_run`), the test set (`assignments`:
sample id, OpenAlex work id, period, and the SHA-256 of each abstract as it was coded), the human coding
(`human_file`: one row per row of the coder's sheet, with its original row number; sample ids filled down, and
`sample_id_filled` marking those that were) and the adjudication decisions.

Abstract text is not in the repository (publisher rights). Each scored abstract is fetched by work id from the
OpenAlex API and rebuilt from `abstract_inverted_index` as results/corpus_pull.py does; no key is needed for
these lookups, but OPENALEX_API_KEY is used if set. Settings are under `abstracts` in the config's `evaluation`
section. Two exceptions, for abstracts whose OpenAlex text changed after coding: CC BY abstracts are read from
`stored` (with licence and DOI), and samples listed under `europepmc` are fetched from that Europe PMC record
(markup removed, whitespace collapsed). Every abstract must match its hash; the script stops and lists any that
do not. Verified abstracts are cached in `cache` (gitignored) and checked again on every run.

Matching. Each human evidence span is located in its abstract: exact match first, then with whitespace and
quotes normalised (nothing else). With `ellipsis_rule` on, an ellipsis at the start or end of a span is trimmed,
and a span joined by an ellipsis is split into pieces, located if every piece is found. Spans still not found
are logged as unlocated and left out. Abstracts in `exclude_samples` are dropped from both sides. Model spans
are located as validate.py does. A model and a human component match if their spans overlap by at least
`min_overlap` of the shorter span's length (a split span: summed over its pieces); each component matches at
most one other, largest overlap first.

Candidate missed matches. Components still unpaired after the overlap matching are compared by label: labels
are embedded with `label_model` (a sentence-embedding model, run locally) and each unpaired human component is
listed with its most similar unpaired model component in the same abstract, if the cosine similarity is at
least `label_min_similarity`. These candidates do not change the scores.

Writes to out_dir: scores.md and confusion_automatic.csv / confusion_adjudicated.csv (no abstract text, no human labels or evidence), and, local
only, side_by_side.md (abstract text and both codings) and the adjudication sheet (`adjudication_file`): every
overlap pair whose types differ (group a) and every label candidate (group b), with a blank Y/N column. An
existing sheet is never overwritten, since it may hold decisions.

Adjudication. If `adjudication_decisions` exists (pair id, row group, Y/N; no labels or evidence), group a pairs
marked N are unpaired and group b candidates marked Y are paired. scores.md reports two tests, identification
and classification, for the automatic and the adjudicated matching side by side.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

import openpyxl
import pandas as pd
import requests

from src.analysis.common import load_config
from src.validate import TYPES, _QUOTES, find_spans, normalise

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
from corpus_pull import BASE, get, rebuild_abstract  # noqa: E402

LOW = ("medium", "low")
ELLIPSIS = re.compile(r"\s*(?:\.{2,}|…)\s*")
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def bin_of(bins):
    return {p: b for b, periods in bins.items() for p in periods}


def read_human(path):
    """Released coding -> list of dicts (row, sample_id, filled_id, type, evidence, label)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return [{"row": int(r.row), "sample_id": r.sample_id, "filled_id": r.sample_id_filled == "Y",
             "type": r.type.strip().lower(), "evidence": r.evidence, "label": r.label} for r in df.itertuples()]


def europepmc_abstract(ref):
    """Abstract of the Europe PMC record ref ("SRC:ID"), markup removed and whitespace collapsed. None if absent."""
    src, rid = ref.split(":")
    r = requests.get(EUROPEPMC, params={"query": f"EXT_ID:{rid} AND SRC:{src}", "resultType": "core",
                                        "format": "json"}, timeout=60)
    r.raise_for_status()
    hits = r.json()["resultList"]["result"]
    text = hits[0].get("abstractText", "") if hits else ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip() or None


def load_abstracts(sample, acfg, exclude):
    """sample_id -> abstract text exactly as coded, for every scored sample. Exits if any fails its hash."""
    stored = pd.read_csv(acfg["stored"], dtype=str, keep_default_na=False).set_index("id")["abstract"]
    europepmc = acfg.get("europepmc") or {}
    os.makedirs(acfg["cache"], exist_ok=True)
    texts, bad = {}, []
    for sid, s in sample.iterrows():
        if sid in exclude:
            continue
        path = os.path.join(acfg["cache"], f"{sid}.txt")
        cached = os.path.exists(path)
        if s.doc_id in stored.index:
            text, src = stored[s.doc_id], acfg["stored"]
        elif cached:
            text, src = open(path, encoding="utf-8", newline="").read(), path
        elif sid in europepmc:
            text, src = europepmc_abstract(europepmc[sid]), f"Europe PMC {europepmc[sid]}"
        else:
            w = get(f"{BASE}/{s.doc_id}", {"select": "id,abstract_inverted_index"})
            text, src = rebuild_abstract(w.get("abstract_inverted_index")), f"OpenAlex {s.doc_id}"
        if text is None or hashlib.sha256(text.encode("utf-8")).hexdigest() != s.abstract_sha256:
            bad.append(f"{sid} ({src}: {'no abstract' if text is None else 'text differs from the coded abstract'})")
            continue
        if not cached and s.doc_id not in stored.index:
            open(path, "w", encoding="utf-8", newline="").write(text)
        texts[sid] = text
    if bad:
        sys.exit("Abstracts not matching their abstract_sha256:\n  " + "\n  ".join(bad))
    return texts


def locate_human(evidence, abstract):
    """(start, end) of the first occurrence: exact, then whitespace and quotes normalised. None if not found."""
    if evidence and evidence in abstract:
        i = abstract.find(evidence)
        return i, i + len(evidence)
    norm, index = normalise(abstract, _QUOTES)
    spans = find_spans(evidence, norm, index, _QUOTES)
    return spans[0] if spans else None


def locate_pieces(evidence, abstract):
    """Ellipsis rule: trim ellipses at the ends, split at inner ones. ([spans], how) or (None, how)."""
    pieces = [p for p in ELLIPSIS.split(evidence.strip()) if p.strip()]
    how = "split" if len(pieces) > 1 else "trimmed"
    spans = [locate_human(p, abstract) for p in pieces]
    return (spans if spans and all(spans) else None), how


def locate_model(evidence, abstract):
    norm, index = normalise(abstract)
    spans = find_spans(evidence, norm, index)
    return spans[0] if spans else None


def overlap(h, m):
    """Characters of span lists h and m that overlap, and the shorter total length."""
    ov = sum(max(0, min(a[1], b[1]) - max(a[0], b[0])) for a in h for b in m)
    return ov, min(sum(e - s for s, e in h), sum(e - s for s, e in m))


def match(human, model, min_overlap):
    """Greedy one-to-one matching, largest overlap first. Sets h['match'] / m['match'] to the other's index."""
    pairs = []
    for i, h in enumerate(human):
        for j, m in enumerate(model):
            ov, shorter = overlap(h["spans"], m["spans"])
            if ov > 0 and ov >= min_overlap * shorter:
                pairs.append((ov, i, j))
    for c in human + model:
        c["match"] = None
    for ov, i, j in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if human[i]["match"] is None and model[j]["match"] is None:
            human[i]["match"], model[j]["match"] = j, i


def kappa(conf):
    n = conf.to_numpy().sum()
    if not n:
        return float("nan")
    po = conf.to_numpy().trace() / n
    pe = (conf.sum(axis=1).to_numpy() * conf.sum(axis=0).to_numpy()).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def measures(abstracts):
    """Measures 1-5 over a list of scored abstracts."""
    H = [h for a in abstracts for h in a["human"]]
    M = [m for a in abstracts for m in a["model"]]
    pairs = [(h["type"], a["model"][h["match"]]["type"]) for a in abstracts for h in a["human"]
             if h["match"] is not None]
    conf = pd.DataFrame(0, index=list(TYPES), columns=list(TYPES))
    for ht, mt in pairs:
        conf.loc[ht, mt] += 1
    share = lambda k, n: (k / n if n else float("nan"), k, n)
    r = {"n_abstracts": len(abstracts), "n_human": len(H), "n_model": len(M), "n_pairs": len(pairs),
         "coverage": share(sum(h["match"] is not None for h in H), len(H)),
         "model_matched": share(sum(m["match"] is not None for m in M), len(M)),
         "same_type": share(sum(a == b for a, b in pairs), len(pairs)),
         "kappa": kappa(conf), "confusion": conf, "per_type": {}, "by_conf": {},
         "kappa_by_type": {t: kappa(pd.DataFrame([[sum(a == t and b == t for a, b in pairs),
                                                   sum(a == t and b != t for a, b in pairs)],
                                                  [sum(a != t and b == t for a, b in pairs),
                                                   sum(a != t and b != t for a, b in pairs)]]))
                           for t in TYPES}}
    for t in TYPES:
        ht, mt = [h for h in H if h["type"] == t], [m for m in M if m["type"] == t]
        r["per_type"][t] = (share(sum(h["match"] is not None for h in ht), len(ht)),
                            share(sum(m["match"] is not None for m in mt), len(mt)))
    for name, levels in (("high", ("high",)), ("medium/low", LOW)):
        ms = [(a, m) for a in abstracts for m in a["model"] if m["confidence"] in levels]
        matched = [(a, m) for a, m in ms if m["match"] is not None]
        r["by_conf"][name] = (share(len(matched), len(ms)),
                              share(sum(a["human"][m["match"]]["type"] == m["type"] for a, m in matched),
                                    len(matched)))
    return r


def pct(s):
    v, k, n = s
    return "–" if not n else f"{100 * v:.0f}% ({k}/{n})"


def scores_md(ecfg, auto, adj, removed, added, log):
    """Two tests, automatic and (if decisions exist) adjudicated matching side by side."""
    who = ecfg["human_label"]
    excl, ellipsis = ecfg.get("exclude_samples") or {}, ecfg.get("ellipsis_rule", False)
    cols = [("automatic", auto)] + ([("adjudicated", adj)] if adj else [])
    head = " | ".join(n for n, _ in cols)
    sep = "|---" * (len(cols) + 1) + "|"
    row = lambda name, f: f"| {name} | " + " | ".join(f(g) for _, g in cols) + " |"
    a = auto["overall"]
    L = [f"# Model vs human coding on the test set", "",
         "_Generated by `evaluation/score_coding.py`; numbers update on every run._", "",
         f"- **Model:** frozen prompt ({log['prompt']}, {log['codebook']}), output in `{ecfg['model_run']}`.",
         f"- **Human coder:** {who}, coding with {ecfg['human_codebook']}.",
         f"- **Abstracts:** {a['n_abstracts']}, time bins from the corpus `period` column." +
         (" Dropped from both sides: " + "; ".join(
             f"{', '.join(k for k in excl if excl[k] == why)} ({why})" for why in dict.fromkeys(excl.values())) + "."
          if excl else ""), "",
         "## Matching and adjudication", "",
         f"Each {who} evidence span is located in its abstract by exact match, then with whitespace and quotes "
         "normalised" + ("; an ellipsis at the start or end of a span is trimmed, and a span joined by an ellipsis "
                         "is split into pieces, located only if every piece is found" if ellipsis else "") +
         f"; spans still not found are left out. A model and a {who} component are paired automatically if their "
         f"evidence spans overlap by at least {100 * ecfg['min_overlap']:.0f}% of the shorter span's length (for a "
         "split span, summed over its pieces), each component pairing at most once, largest overlap first. For "
         "adjudication, two kinds of pair were reviewed by hand: every automatic pair whose types differ, and, for "
         f"components left unpaired, each {who} component's most similar unpaired model component in the same "
         f"abstract by label (sentence-embedding cosine similarity of at least {ecfg['label_min_similarity']}). "
         "Automatic pairs judged not the same component are unpaired; label candidates judged the same component "
         "are paired. Decisions are in `adjudication_decisions.csv` (pair ids and Y/N only).", ""]
    if adj:
        d = adj["overall"]
        L += [f"**Pairs:** {a['n_pairs']} automatic, {d['n_pairs']} adjudicated "
              f"({removed} removed, {added} added).", ""]

    L += ["## Test 1: identification", "",
          f"Does the model find the components {who} coded, and does {who} code the ones the model finds? "
          f"Components left out below cannot pair, so the second measure is a lower bound.", "",
          f"| | {head} |", sep,
          row(f"coverage: {who}'s components found by the model, overall", lambda g: pct(g["overall"]["coverage"])),
          row(f"model's components also coded by {who}, overall", lambda g: pct(g["overall"]["model_matched"]))]
    for b in ecfg["bins"]:
        L += [row(f"coverage, {b}", lambda g: pct(g[b]["coverage"])),
              row(f"model's also coded, {b}", lambda g: pct(g[b]["model_matched"]))]
    L += ["", "By type (coverage by the human type; model's also coded by the model type):", "",
          f"| | {head} |", sep]
    for k, t in enumerate(TYPES):
        L += [row(f"{t}: coverage", lambda g: pct(g["overall"]["per_type"][t][0])),
              row(f"{t}: model's also coded", lambda g: pct(g["overall"]["per_type"][t][1]))]
    L += ["", "By the model's confidence (share of its components also coded):", "", f"| | {head} |", sep]
    L += [row(c, lambda g: pct(g["overall"]["by_conf"][c][0])) for c in ("high", "medium/low")]

    L += ["", "## Test 2: classification", "",
          "On paired components, does the model give the same type? Cohen's kappa overall (four types) and per "
          "type (that type against the rest).", "",
          f"| | {head} |", sep,
          row("pairs", lambda g: str(g["overall"]["n_pairs"])),
          row("same type", lambda g: pct(g["overall"]["same_type"])),
          row("Cohen's kappa, all types", lambda g: f"{g['overall']['kappa']:.2f}")]
    L += [row(f"Cohen's kappa, {t} vs rest", lambda g: f"{g['overall']['kappa_by_type'][t]:.2f}") for t in TYPES]
    L += ["", "By time bin:", "", f"| | {head} |", sep]
    for b in ecfg["bins"]:
        L += [row(f"{b}: same type", lambda g: pct(g[b]["same_type"])),
              row(f"{b}: Cohen's kappa", lambda g: f"{g[b]['kappa']:.2f}")]
    L += ["", "By the model's confidence (type agreement on its paired components):", "", f"| | {head} |", sep]
    L += [row(c, lambda g: pct(g["overall"]["by_conf"][c][1])) for c in ("high", "medium/low")]
    for name, g in cols:
        conf = g["overall"]["confusion"]
        L += ["", f"Confusion, {name} pairs ({who} rows × model columns):", "",
              "| " + who + " \\ model | " + " | ".join(TYPES) + " | total |", "|---" * (len(TYPES) + 2) + "|"]
        L += [f"| {t} | " + " | ".join(str(conf.loc[t, c]) for c in TYPES) + f" | {conf.loc[t].sum()} |"
              for t in TYPES]
        L.append("| total | " + " | ".join(str(conf[c].sum()) for c in TYPES) + f" | {conf.to_numpy().sum()} |")

    L += ["", f"## {who} components used", "",
          "| | n |", "|---|---|",
          f"| Rows in the coding sheet | {log['rows']} |",
          f"| …of which sample id filled down from the row above | {log['filled_id']} |",
          f"| Dropped with their abstracts ({log['excluded_abstracts']} abstracts not scored; the model's "
          f"{log['excluded_model']} components there are dropped too) | {log['excluded_rows']} |",
          f"| Left out: type not one of the four (e.g. \"Unsure\") | {log['bad_type']} |",
          f"| Left out: no evidence given | {log['no_evidence']} |",
          f"| Left out: evidence not located in the abstract | {log['unlocated']} |",
          f"| Located, used for scoring | {log['located']} |", "",
          f"Located by exact match: {log['exact']}; after normalising whitespace and quotes: {log['normalised']}" +
          (f"; after trimming an ellipsis at the start or end: {log['trimmed']}; after splitting at an ellipsis: "
           f"{log['split']}" if ellipsis else "") +
          f". Model components not located in the abstract: {log['model_unlocated']} (of {log['model_total']}).", ""]
    return "\n".join(L)


def side_by_side_md(ecfg, scored):
    who = ecfg["human_label"]
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ").strip()
    L = [f"# Side by side: {who} vs model (test set)", "",
         "_Local only (abstract text and human coding). Generated by `python -m src.analysis.score_coding`._", "",
         "`#` is the match: the same number on both sides marks a matched pair; ✗ marks a pair whose types "
         "differ. Rows marked *left out* were not scored.", ""]
    for a in scored:
        L += [f"## {a['sample_id']} · {a['work_id']} · {a['year']} · {a['bin']}", "", f"**{a['title']}**", "",
              a["abstract"], "", f"**{who}**", "", "| # | type | label | evidence |", "|---|---|---|---|"]
        for h in a["human"]:
            if h["match"] is None:
                k = ""
            else:
                k = str(h["pair"]) + ("" if a["model"][h["match"]]["type"] == h["type"] else " ✗")
            L.append(f"| {k} | {h['type']} | {esc(h['label'])} | {esc(h['evidence'])} |")
        for h in a["left_out"]:
            L.append(f"| *left out: {h['why']}* | {h['type']} | {esc(h['label'])} | {esc(h['evidence'])} |")
        L += ["", "**Model**", "", "| # | type | label | evidence | confidence |", "|---|---|---|---|---|"]
        for m in a["model"]:
            k = "" if m["match"] is None else str(a["human"][m["match"]]["pair"])
            L.append(f"| {k} | {m['type']} | {esc(m['label'])} | {esc(m['evidence'])} | {m['confidence']} |")
        L.append("")
    return "\n".join(L)


def label_candidates(scored, model_name, threshold):
    """Per abstract: each unpaired human component with its most similar unpaired model component (by label)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    out = []
    for a in scored:
        hs = [h for h in a["human"] if h["match"] is None and h["label"].strip()]
        ms = [(j, m) for j, m in enumerate(a["model"]) if m["match"] is None and m["label"].strip()]
        if not hs or not ms:
            continue
        eh = model.encode([h["label"] for h in hs], normalize_embeddings=True)
        em = model.encode([m["label"] for _, m in ms], normalize_embeddings=True)
        sim = eh @ em.T
        for i, h in enumerate(hs):
            k = int(sim[i].argmax())
            if sim[i, k] >= threshold:
                out.append((a, h, ms[k][0], float(sim[i, k])))
    return out


def pair_id(a, h, j):
    """Stable id of a human-model pair: sample, coding-sheet row, model component number in that abstract."""
    return f"{a['sample_id']}:r{h['row']}:m{j + 1}"


def write_adjudication(path, scored, candidates, who):
    """Group a: overlap pairs whose types differ. Group b: label candidates. Blank decision column."""
    if os.path.exists(path):
        return None
    rows = []
    for a in scored:
        for h in a["human"]:
            if h["match"] is not None and a["model"][h["match"]]["type"] != h["type"]:
                rows.append((a, h, h["match"], "a", None))
    rows += [(a, h, j, "b", round(sim, 3)) for a, h, j, sim in candidates]
    rows.sort(key=lambda r: (r[0]["sample_id"], r[3], r[1]["row"]))
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Adjudication"
    ws.append(["Sample", "Pair id", "Row group", "Similarity (b)", f"{who} type", f"{who} label", f"{who} evidence",
               "Model type", "Model label", "Model evidence", "Same component? (Y/N)"])
    for a, h, j, group, sim in rows:
        m = a["model"][j]
        ws.append([a["sample_id"], pair_id(a, h, j), group, sim, h["type"], h["label"], h["evidence"],
                   m["type"], m["label"], m["evidence"], None])
    for col, w in zip("ABCDEFGHIJK", (8, 16, 10, 12, 12, 30, 50, 12, 30, 50, 14)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    wb.save(path)
    return Counter(r[3] for r in rows)


def adjudicate(scored, path):
    """Copy of scored with group a pairs marked N unpaired and group b pairs marked Y paired.

    Returns (abstracts, n removed, n added). Exits if a decision does not fit the automatic matching."""
    dec = pd.read_csv(path, dtype=str)
    out, by_id = [], {}
    for a in scored:
        c = {**a, "human": [dict(h) for h in a["human"]], "model": [dict(m) for m in a["model"]]}
        out.append(c)
        for i, h in enumerate(c["human"]):
            for j in range(len(c["model"])):
                by_id[pair_id(c, h, j)] = (c, i, j)
    removed = added = 0
    for r in dec.itertuples():
        if r.pair_id not in by_id:
            sys.exit(f"{path}: pair {r.pair_id} not found in the current scoring")
        c, i, j = by_id[r.pair_id]
        h, m = c["human"][i], c["model"][j]
        if r.row_group == "a" and r.same_component == "N":
            if h["match"] != j or m["match"] != i:
                sys.exit(f"{path}: group a pair {r.pair_id} is not an automatic pair")
            h["match"] = m["match"] = None
            removed += 1
    for r in dec.itertuples():
        c, i, j = by_id[r.pair_id]
        h, m = c["human"][i], c["model"][j]
        if r.row_group == "b" and r.same_component == "Y":
            if h["match"] is not None or m["match"] is not None:
                sys.exit(f"{path}: group b pair {r.pair_id} has a component that is already paired")
            h["match"], m["match"] = j, i
            added += 1
    return out, removed, added


def score(sample, texts, outputs, rows, bins, ecfg, ellipsis_rule, exclude):
    """Locate and match every abstract in the sample. Returns (scored abstracts, log counts)."""
    log = Counter(rows=len(rows), filled_id=sum(r["filled_id"] for r in rows))
    scored = []
    for sid, s in sample.iterrows():
        if sid in exclude:
            log["excluded_abstracts"] += 1
            log["excluded_rows"] += sum(r["sample_id"] == sid for r in rows)
            log["excluded_model"] += len(outputs.get(s.doc_id, {"components": []})["components"])
            continue
        abstract = texts.loc[sid, "abstract"]
        a = {"sample_id": sid, "work_id": s.doc_id, "year": s.year, "bin": bins[s.period],
             "title": texts.loc[sid, "title"], "abstract": abstract, "human": [], "left_out": [], "model": []}
        for r in (r for r in rows if r["sample_id"] == sid):
            if r["type"] not in TYPES:
                log["bad_type"] += 1
                a["left_out"].append({**r, "why": "type"})
            elif not r["evidence"].strip():
                log["no_evidence"] += 1
                a["left_out"].append({**r, "why": "no evidence"})
            else:
                span = locate_human(r["evidence"], abstract)
                spans, how = ([span], "exact" if r["evidence"] in abstract else "normalised") if span else (None, "")
                if spans is None and ellipsis_rule and ELLIPSIS.search(r["evidence"]):
                    spans, how = locate_pieces(r["evidence"], abstract)
                if spans is None:
                    log["unlocated"] += 1
                    log["unlocated_ellipsis"] += bool(ELLIPSIS.search(r["evidence"]))
                    a["left_out"].append({**r, "why": "unlocated"})
                else:
                    log["located"] += 1
                    log[how] += 1
                    a["human"].append({**r, "spans": spans, "how": how})
        for c in outputs.get(s.doc_id, {"components": []})["components"]:
            log["model_total"] += 1
            span = locate_model(c["evidence"], abstract)
            if span is None:
                log["model_unlocated"] += 1
            else:
                a["model"].append({**c, "spans": [span]})
        match(a["human"], a["model"], ecfg["min_overlap"])
        for k, h in enumerate((h for h in a["human"] if h["match"] is not None), start=1):
            h["pair"] = k
        scored.append(a)
    return scored, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg, _ = load_config(args.config)
    ecfg = cfg["evaluation"]
    out = ecfg["out_dir"]
    os.makedirs(out, exist_ok=True)

    sample = pd.read_csv(ecfg["assignments"])
    sample = sample[sample.set == ecfg["sample_set"]].set_index("sample_id")
    exclude = ecfg.get("exclude_samples") or {}
    texts = pd.DataFrame({"title": sample.title,
                          "abstract": pd.Series(load_abstracts(sample, ecfg["abstracts"], exclude), dtype=object)})
    outputs = {}
    with open(os.path.join(ecfg["model_run"], "outputs.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            outputs[o["work_id"]] = o
    run_log = json.load(open(os.path.join(ecfg["model_run"], "run_log.json"), encoding="utf-8"))
    rows = read_human(ecfg["human_file"])
    bins = bin_of(ecfg["bins"])

    scored, log = score(sample, texts, outputs, rows, bins, ecfg, ecfg.get("ellipsis_rule", False), exclude)
    stray = sorted({r["sample_id"] for r in rows} - set(sample.index), key=str)
    if stray:
        print(f"warning: coding rows for sample ids not in the sample set: {stray}")

    by_bin = lambda sc: {"overall": measures(sc), **{b: measures([a for a in sc if a["bin"] == b])
                                                     for b in ecfg["bins"]}}
    groups = by_bin(scored)
    log["prompt"] = os.path.basename(run_log.get("prompt_file", "")) or "prompt"
    log["codebook"] = os.path.basename(run_log.get("codebook_file", "")) or "codebook"

    open(os.path.join(out, "side_by_side.md"), "w", encoding="utf-8").write(side_by_side_md(ecfg, scored))
    candidates = label_candidates(scored, ecfg["label_model"], ecfg["label_min_similarity"])
    sheet = write_adjudication(ecfg["adjudication_file"], scored, candidates, ecfg["human_label"])
    adj, removed, added = None, 0, 0
    if ecfg.get("adjudication_decisions") and os.path.exists(ecfg["adjudication_decisions"]):
        adj, removed, added = adjudicate(scored, ecfg["adjudication_decisions"])
        adj = by_bin(adj)
    open(os.path.join(out, "scores.md"), "w", encoding="utf-8").write(
        scores_md(ecfg, groups, adj, removed, added, log))
    for name, g in [("automatic", groups)] + ([("adjudicated", adj)] if adj else []):
        conf = g["overall"]["confusion"].rename_axis(f"{ecfg['human_label']} \\ model")
        conf.to_csv(os.path.join(out, f"confusion_{name}.csv"))

    print({k: log[k] for k in ("rows", "filled_id", "excluded_abstracts", "excluded_rows", "bad_type",
                               "no_evidence", "unlocated", "unlocated_ellipsis", "located", "exact", "normalised",
                               "trimmed", "split", "model_total", "model_unlocated")})
    if sheet is None:
        print(f"adjudication sheet {ecfg['adjudication_file']} exists: left as is")
    else:
        print(f"adjudication sheet {ecfg['adjudication_file']}: {sheet['a'] + sheet['b']} rows "
              f"(a: {sheet['a']} overlap pairs with different types; b: {sheet['b']} label candidates)")
    for name, g in [("automatic", groups)] + ([("adjudicated", adj)] if adj else []):
        r = g["overall"]
        print(f"{name}: pairs {r['n_pairs']}; coverage {pct(r['coverage'])}; model matched {pct(r['model_matched'])}; "
              f"same type {pct(r['same_type'])}; kappa {r['kappa']:.2f}; by type "
              + ", ".join(f"{t} {r['kappa_by_type'][t]:.2f}" for t in TYPES))
    if adj:
        print(f"adjudication: {removed} pairs removed, {added} added")


if __name__ == "__main__":
    main()
