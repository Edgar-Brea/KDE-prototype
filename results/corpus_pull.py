#!/usr/bin/env python3
"""
KDE · corpus_pull.py — pull CRISPR + antecedent abstracts from OpenAlex.

  python results/corpus_pull.py --dry-run   # counts only (fast) -> check the query terms first
  python results/corpus_pull.py    # full pull -> data/corpus.parquet + data/corpus_counts.md

HOW IT SAVES YOUR OPENALEX CREDITS
  One full pull is enough. Every page of results is saved to disk as it arrives
  (data/raw/), with a progress marker. So:
  - If the daily free allowance runs out mid-pull, the script stops, keeps what it has,
    and the same command tomorrow resumes from where it stopped.
  - Rerunning after a finished pull downloads nothing; it rebuilds the corpus from data/raw/.
  - Dropping a search later = delete it from QUERIES and rerun: no download, it's left out.
    (Adding or editing a search downloads only that search.)
  - Full OpenAlex records are kept in data/raw/, so later needs (authors, references)
    don't require pulling again.
  --dry-run is optional: ~25 calls (about 3 cents of the ~$1/day free allowance).

OUTPUTS (paths are relative to the folder you run the script from):
  --dry-run   Writes NO files. Prints to screen only: paper counts per query x period
              (total / with abstract), and each seed DOI with the title it resolves to.
  full pull   data/raw/<query>.jsonl.gz  full OpenAlex records per search (the saved download)
              data/raw/seeds.json        the landmark papers looked up by DOI
              data/corpus.parquet    one row per paper: doc_id, doi, title, abstract, year,
                                     date, venue, type, language, cited_by, matched (which
                                     queries found it), period. Falls back to data/corpus.csv
                                     if pyarrow isn't installed.
              data/corpus_counts.md  counts by period (with % missing abstracts), counts by
                                     query, seed DOIs not found, and the exact queries used.
  Run from the repo root, or change the folder with --out.

Needs: requests, pandas (pyarrow optional; falls back to CSV).
Env vars: OPENALEX_API_KEY (required, free from openalex.org/settings/api); OPENALEX_MAILTO optional.

Kept for provenance: this is the script that built the CRISPR corpus. It pulls everything the
searches return; eligibility (abstract length) is applied later, at extraction time.
"""
import argparse, gzip, os, re, sys, time, json
import requests
import pandas as pd

BASE = "https://api.openalex.org/works"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")
API_KEY = os.environ.get("OPENALEX_API_KEY", "")

# ===================== SETTINGS =====================
# Period boundaries (a sampling device: they make sure old papers reach the test set)
PERIODS = [("antecedent", None, 2001), ("early", 2002, 2011), ("post", 2012, None)]

# Each query searches titles + abstracts. Years are inclusive; None = open-ended.
QUERIES = {
    # Main CRISPR literature
    "crispr": dict(
        expr='"CRISPR" OR "clustered regularly interspaced short palindromic repeats"',
        y0=None, y1=None),
    # Pre-standardised names for the repeat loci
    "antecedent_named": dict(
        expr='"short regularly spaced repeats" OR "regularly spaced repeats" OR '
             '"spacers interspersed direct repeats" OR "large clusters of tandem repeats"',
        y0=None, y1=2011),
    # Generic prokaryotic repeat vocabulary. Noisy (IS elements, REP, VNTR typing); see FILTERS.
    "antecedent_generic": dict(
        expr='("direct repeats" OR "tandem repeats" OR "repetitive sequences") AND '
             '(archaea OR archaeal OR prokaryotic OR "bacterial genome")',
        y0=None, y1=2005),
    # Bacterial-immunity side: dairy phage-resistance work that fed Barrangou 2007.
    "antecedent_phage": dict(
        expr='"phage resistance" AND ("Streptococcus thermophilus" OR "lactic acid bacteria")',
        y0=None, y1=2007),
    # M. tuberculosis DR locus typing. Real antecedent vocabulary but mostly epidemiology.
    "antecedent_spoligo": dict(
        expr='"spoligotyping" OR "direct repeat locus" OR "DR locus"',
        y0=None, y1=2005),
}

# Landmark papers forced into the corpus whatever the queries return.
# The script reports any DOI that fails to resolve.
SEED_DOIS = {
    "Ishino 1987 (iap gene, E. coli repeats)":        "10.1128/jb.169.12.5429-5433.1987",
    "Mojica 2000 (regularly spaced repeats)":         "10.1046/j.1365-2958.2000.01838.x",
    "Jansen 2002 (names CRISPR, cas genes)":          "10.1046/j.1365-2958.2002.02839.x",
    "Mojica 2005 (spacers from extrachromosomal)":    "10.1007/s00239-004-0046-3",
    "Pourcel 2005 (spacers from phage)":              "10.1099/mic.0.27437-0",
    "Bolotin 2005 (spacers, S. thermophilus)":        "10.1099/mic.0.28048-0",
    "Makarova 2006 (putative RNAi-like immunity)":    "10.1186/1745-6150-1-7",
    "Barrangou 2007 (CRISPR adaptive immunity)":      "10.1126/science.1138140",
    "Jinek 2012 (programmable Cas9)":                 "10.1126/science.1225829",
}

DOC_TYPES = "article|review|preprint"
PER_PAGE = 100   # OpenAlex maximum per page

# The term "CRISPR" was coined in 2002 (Jansen et al.). A paper dated earlier whose text
# mentions it is an OpenAlex error (an abstract attached to the wrong record, or a wrong
# year) and is dropped. Seeds are exempt. Example: a 1977 cardiology paper carrying a
# CRISPR abstract.
CRISPR_COINED = 2002

# Same idea for genome editing: programmable Cas9 dates from 2012 (Jinek et al.), so a paper
# dated earlier that uses editing vocabulary is a record error (21 of 521 early-period papers,
# nearly all medical letters carrying modern abstracts).
EDITING_FROM = 2012
EDITING_TERMS = r"genome editing|gene editing|\bsgRNAs?\b|single guide RNA|base edit|prime edit|CRISPR[-/]Cas9"

# Papers reviewed by hand and kept although a removal rule (anachronism / off-topic title)
# would drop them. OpenAlex ID -> reason. Reviewed by hand from the off-topic-title list.
KEEP = {
    "W776872394":  "2008 Chinese-language CRISPR review (title not in English)",
    "W2008344040": "2011 Polish review of phage resistance in lactic acid bacteria",
    "W2064146176": "2011 Haloquadratum walsbyi genomics (archaeal; reports CRISPR loci)",
    "W2097146199": "2011 human gut virome study (CRISPR spacers in gut bacteria)",
    "W2142813640": "2011 early-branching S. aureus lineage genome",
    "W628414444":  "2009 molecular epidemiology methods book (typing methods)",
}

# A paper dated before 2012 found ONLY by the CRISPR search whose TITLE has no biology word
# is taken to be a record carrying someone else's abstract (e.g. "Focal hand dystonia").
BIO_TITLE = (r"crispr|\bcas\d*\b|cas\d|repeat|spacer|phage|vir(us|al)|bacteri|archae|prokaryot|microb|"
             r"\bgen(e|es|etic|etics|ome|omes|omic|omics)\b|\bdna\b|\brna|immun|strain|sequenc|plasmid|protein|"
             r"enzyme|nuclease|tubercul|mycobact|streptococ|thermophil|escherichia|salmonella|yersinia|"
             r"evolution|defen[cs]e|resistan|transcri|helicase|locus|loci|operon|typing|polymorph|interference|"
             r"pathogen|isolate|species|clone|clonal|chromosom|mobile element")

# Relevance filters, applied locally to the saved downloads (free; no re-download).
# A paper found by a search is kept only if its title or abstract matches that search's
# pattern (case-insensitive). A paper found by several searches is kept if ANY of them
# keeps it. Seeds are always kept. Set a pattern to None to switch its filter off.
# Set after inspecting samples of each search.
FILTERS = {
    # OpenAlex returned papers that never mention CRISPR (e.g. 1980s Annual Reviews).
    "crispr": r"crispr|clustered regularly interspaced",
    # Early CRISPR vocabulary only; the rest of this search was unrelated repeat biology.
    # A list = every pattern must match (here: early vocabulary AND a microbial context;
    # "regularly spaced" alone let through a 1977 cardiology paper).
    "antecedent_generic": [r"regularly spaced|\bSRSRs?\b|\bSPIDRs?\b|\bLCTRs?\b",
                           r"prokaryot|archae|bacteri"],
    # "DR locus" also matched HLA-DR immunology; keep only the M. tuberculosis DR locus.
    "antecedent_spoligo": r"spoligo|tubercul|mycobacter",
}
# ============================================================

class OutOfCredits(Exception):
    pass


def auth():
    return {k: v for k, v in (("mailto", MAILTO), ("api_key", API_KEY)) if v}


def get(url, params, tries=6):
    params = dict(params, **auth())
    for i in range(tries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            sys.exit(f"OpenAlex refused the request ({r.status_code}). Check OPENALEX_API_KEY.\n{r.text[:300]}")
        if r.status_code == 429 and "credit" in r.text.lower():
            raise OutOfCredits(r.text[:200])          # daily allowance used up: stop, don't retry
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** i); continue
        sys.exit(f"HTTP {r.status_code} for {r.url}\n{r.text[:300]}")
    sys.exit(f"Gave up after {tries} retries: {url}")


def year_filter(y0, y1):
    if y0 and y1: return f"publication_year:{y0}-{y1}"
    if y0: return f"publication_year:>{y0 - 1}"
    if y1: return f"publication_year:<{y1 + 1}"
    return None


def build_filter(expr, y0=None, y1=None, extra=None):
    parts = [f"title_and_abstract.search:{expr}", f"type:{DOC_TYPES}"]
    yf = year_filter(y0, y1)
    if yf: parts.append(yf)
    if extra: parts.append(extra)
    return ",".join(parts)


def clip(qy0, qy1, py0, py1):
    """Intersect a query's year range with a period's; None if empty."""
    lo = max(x for x in (qy0, py0) if x is not None) if (qy0 or py0) else None
    hi = min(x for x in (qy1, py1) if x is not None) if (qy1 or py1) else None
    if lo and hi and lo > hi: return "empty"
    return lo, hi


def rebuild_abstract(inv):
    if not inv: return None
    pos = {}
    for word, idxs in inv.items():
        for i in idxs: pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def period_of(year):
    if year is None: return None
    for name, y0, y1 in PERIODS:
        if (y0 is None or year >= y0) and (y1 is None or year <= y1):
            return name


def to_row(w):
    src = ((w.get("primary_location") or {}).get("source") or {})
    return dict(
        doc_id=w["id"].rsplit("/", 1)[-1],
        doi=(w.get("doi") or "").replace("https://doi.org/", "") or None,
        title=w.get("title"),
        abstract=rebuild_abstract(w.get("abstract_inverted_index")),
        year=w.get("publication_year"),
        date=w.get("publication_date"),
        venue=src.get("display_name"),
        type=w.get("type"),
        language=w.get("language"),
        cited_by=w.get("cited_by_count"),
    )


def dry_run():
    print("Counts per query x period (total / with abstract). No data downloaded.\n")
    lines = ["| query | period | total | with abstract |", "|---|---|---|---|"]
    for qn, q in QUERIES.items():
        for pn, py0, py1 in PERIODS:
            c = clip(q["y0"], q["y1"], py0, py1)
            if c == "empty": continue
            f = build_filter(q["expr"], *c)
            tot = get(BASE, {"filter": f, "per-page": 1})["meta"]["count"]
            ab = get(BASE, {"filter": f + ",has_abstract:true", "per-page": 1})["meta"]["count"]
            line = f"| {qn} | {pn} | {tot} | {ab} |"
            lines.append(line); print(line)
    print("\nSeed DOIs:")
    for label, doi in SEED_DOIS.items():
        try:
            w = requests.get(f"{BASE}/doi:{doi}", params=auth(),
                             timeout=30)
            ok = w.status_code == 200
            has_ab = ok and bool(w.json().get("abstract_inverted_index"))
            got = f"-> {w.json().get('publication_year')} {(w.json().get('title') or '')[:90]}" if ok else ""
            print(f"  {'OK ' if ok else 'MISSING'} {'(abstract)' if has_ab else '(no abstract)' if ok else ''}  {label}  {doi}\n      {got}")
        except Exception as e:
            print(f"  ERROR {label}: {e}")
    return lines


def pull_query(qn, q, rawdir):
    """Download one query into data/raw/<qn>.jsonl.gz, page by page.
    Progress is saved after every page, so an interrupted pull resumes where it stopped."""
    data, statef = os.path.join(rawdir, f"{qn}.jsonl.gz"), os.path.join(rawdir, f"{qn}.state.json")
    spec = dict(expr=q["expr"], y0=q["y0"], y1=q["y1"], types=DOC_TYPES)
    st = json.load(open(statef)) if os.path.exists(statef) else None
    if st and st["spec"] != spec:                       # query text changed -> start that query afresh
        print(f"  {qn}: query changed since last run, re-downloading")
        st = None
        if os.path.exists(data): os.remove(data)
    st = st or dict(spec=spec, cursor="*", n=0, done=False)
    if st["done"]:
        if "expected" not in st:                        # older download: 1 call to check it's complete
            st["expected"] = get(BASE, {"filter": build_filter(q["expr"], q["y0"], q["y1"]),
                                        "per-page": 1})["meta"]["count"]
            json.dump(st, open(statef, "w"))
        flag = "" if st["n"] >= 0.98 * st["expected"] else "  WARNING: INCOMPLETE"
        print(f"  {qn}: already downloaded ({st['n']} of {st['expected']} records), skipping{flag}"); return
    f = build_filter(q["expr"], q["y0"], q["y1"])
    while True:
        d = get(BASE, {"filter": f, "per-page": PER_PAGE, "cursor": st["cursor"]})
        st["expected"] = d["meta"]["count"]
        with gzip.open(data, "at") as fh:
            for w in d["results"]:
                fh.write(json.dumps(w) + "\n")
        st["n"] += len(d["results"])
        st["cursor"] = d["meta"].get("next_cursor")
        st["done"] = not d["results"] or not st["cursor"]
        json.dump(st, open(statef, "w"))
        print(f"\r  {qn}: {st['n']} / {d['meta']['count']}  (~{-(-d['meta']['count'] // PER_PAGE)} calls total)",
              end="", flush=True)
        if st["done"]:
            json.dump(st, open(statef, "w"))
            print()
            if st["n"] < 0.98 * st["expected"]:
                print(f"  WARNING: {qn} stopped at {st['n']} of {st['expected']} expected records. "
                      f"Check before using this corpus.")
            return


def full_pull(outdir):
    rawdir = os.path.join(outdir, "raw")
    os.makedirs(rawdir, exist_ok=True)
    try:
        for qn, q in QUERIES.items():
            pull_query(qn, q, rawdir)
    except OutOfCredits:
        sys.exit("\n\nDaily OpenAlex allowance used up. Nothing is lost: everything downloaded so far is "
                 "saved in data/raw/.\nRerun the same command after midnight UTC and it"
                 "carries on from where it stopped.")

    # Seeds: DOI lookups are free; cached so they're fetched once.
    seedf = os.path.join(rawdir, "seeds.json")
    seeds = json.load(open(seedf)) if os.path.exists(seedf) else {}
    for label, doi in SEED_DOIS.items():
        if doi in seeds: continue
        r = requests.get(f"{BASE}/doi:{doi}", params=auth(), timeout=30)
        seeds[doi] = r.json() if r.status_code == 200 else None
    json.dump(seeds, open(seedf, "w"))

    # Assemble the corpus from the saved raw files (no network from here on).
    rows, tags = {}, {}
    for qn in QUERIES:                                  # a query deleted from QUERIES is simply left out
        with gzip.open(os.path.join(rawdir, f"{qn}.jsonl.gz"), "rt") as fh:
            for line in fh:
                r = to_row(json.loads(line))
                rows.setdefault(r["doc_id"], r)
                tags.setdefault(r["doc_id"], set()).add(qn)
    missing_seeds = []
    for label, doi in SEED_DOIS.items():
        if not seeds.get(doi):
            missing_seeds.append(f"{label} ({doi})"); continue
        r = to_row(seeds[doi])
        rows.setdefault(r["doc_id"], r)
        tags.setdefault(r["doc_id"], set()).add("seed")

    df = pd.DataFrame(rows.values())
    df["matched"] = df["doc_id"].map(lambda i: "|".join(sorted(tags[i])))
    df["period"] = df["year"].map(period_of)

    # Relevance filters (see FILTERS). Record what each one removes.
    text = (df["title"].fillna("") + " " + df["abstract"].fillna(""))
    def match(p):
        if not p: return pd.Series(True, index=df.index)
        m = pd.Series(True, index=df.index)
        for pat in ([p] if isinstance(p, str) else p):
            m &= text.str.contains(pat, case=False, regex=True)
        return m
    passes = {q: match(p) for q, p in FILTERS.items() if q in QUERIES}
    def keep(i):
        qs = tags[df.at[i, "doc_id"]]
        return "seed" in qs or any(passes[q][i] if q in passes else True for q in qs)
    kept = df.index.map(keep).to_numpy(dtype=bool)
    filt_rows = []
    for q in passes:
        hit = df["matched"].str.contains(q)
        filt_rows.append((q, int(hit.sum()), int((hit & kept).sum())))
    df = df[kept].copy()

    # Anachronisms (see CRISPR_COINED).
    t2 = df["title"].fillna("") + " " + df["abstract"].fillna("")
    notseed = ~df["matched"].str.contains("seed") & ~df["doc_id"].isin(list(KEEP))
    anach = notseed & (
        ((df["year"] < CRISPR_COINED) & t2.str.contains(FILTERS["crispr"], case=False, regex=True)) |
        ((df["year"] < EDITING_FROM) & t2.str.contains(EDITING_TERMS, case=False, regex=True)))
    anach_rows = df.loc[anach, ["year", "doc_id", "title"]]
    df = df[~anach].copy()
    offtopic = (~df["doc_id"].isin(list(KEEP)) & (df["year"] < EDITING_FROM) & (df["matched"] == "crispr")
                & ~df["title"].fillna("").str.contains(BIO_TITLE, case=False, regex=True))
    offtopic_rows = df.loc[offtopic, ["year", "doc_id", "title"]]
    df = df[~offtopic].copy()

    # Second dedup: same normalised title + year (preprint/published twins). Prefer a row with an abstract.
    df["_t"] = df["title"].fillna("").str.lower().map(lambda s: re.sub(r"[^a-z0-9]", "", s))
    df["_has"] = df["abstract"].notna()
    before = len(df)
    df = (df.sort_values(["_has", "cited_by"], ascending=False)
            .loc[lambda x: (x["_t"] == "") | ~x.duplicated(["_t", "year"])]
            .assign(_d=lambda x: x["doi"].str.lower().str.replace(r"\.eor$", "", regex=True))
            .loc[lambda x: x["_d"].isna() | ~x.duplicated("_d")]      # same DOI (".eor" = early-release copy)
            .drop(columns=["_d"])
            .drop(columns=["_t", "_has"]).sort_values(["year", "doc_id"]).reset_index(drop=True))
    twins = before - len(df)
    generic_survivors = df.loc[df["matched"].str.contains("antecedent_generic"), ["year", "title"]]

    os.makedirs(outdir, exist_ok=True)
    try:
        path = os.path.join(outdir, "corpus.parquet"); df.to_parquet(path, index=False)
    except Exception:
        path = os.path.join(outdir, "corpus.csv"); df.to_csv(path, index=False)

    t = df.assign(has_abstract=df["abstract"].notna()).groupby("period", dropna=False).agg(
        total=("doc_id", "size"), with_abstract=("has_abstract", "sum"))
    t["missing_pct"] = (100 * (1 - t["with_abstract"] / t["total"])).round(1)
    order = [p[0] for p in PERIODS]
    t = t.reindex([p for p in order if p in t.index] + [p for p in t.index if p not in order])

    report = [f"# Corpus counts — {time.strftime('%Y-%m-%d')}", "",
              f"File: `{path}` · {len(df)} docs · {twins} duplicates removed (same title+year or same DOI)", "",
              "## By period", "", "| period | total | with abstract | missing % |", "|---|---|---|---|"]
    report += [f"| {i} | {int(r.total)} | {int(r.with_abstract)} | {r.missing_pct} |" for i, r in t.iterrows()]
    report += ["", "## By query (a doc can match several)", "",
               "| query | in corpus | downloaded | OpenAlex expected | complete? |", "|---|---|---|---|---|"]
    for qn in list(QUERIES) + ["seed"]:
        sf = os.path.join(rawdir, f"{qn}.state.json")
        st = json.load(open(sf)) if os.path.exists(sf) else {}
        exp, got = st.get("expected"), st.get("n")
        ok = "" if exp is None else ("yes" if got >= 0.98 * exp else "**NO**")
        report.append(f"| {qn} | {int(df['matched'].str.contains(qn).sum())} | {got if got is not None else ''} "
                      f"| {exp if exp is not None else ''} | {ok} |")
    report += ["", "## Relevance filters (applied locally)", "",
               "| query | found by search | kept after filter | removed |", "|---|---|---|---|"]
    report += [f"| {q} | {a} | {b} | {a - b} |" for q, a, b in filt_rows]
    report += ["", f"## Anachronisms removed: {len(anach_rows)}", "",
               f"Dated before {CRISPR_COINED} but mentioning CRISPR, or before {EDITING_FROM} but using "
               f"genome-editing terms (OpenAlex record errors).", ""]
    old_a = anach_rows[anach_rows["year"] < CRISPR_COINED].sort_values("year")
    mid_a = anach_rows[anach_rows["year"] >= CRISPR_COINED].sort_values("year")
    report += [f"### Before {CRISPR_COINED}: {len(old_a)} (first 25 shown)", ""]
    report += [f"- {int(y)}: {t} ({i})" for y, i, t in old_a.head(25).itertuples(index=False)]
    if len(old_a) > 25: report.append(f"- … and {len(old_a) - 25} more")
    report += ["", f"### {CRISPR_COINED}–{EDITING_FROM - 1}, editing terms: {len(mid_a)} (all shown — check none is a "
               f"genuine paper that mentions genome editing in passing)", ""]
    report += [f"- {int(y)}: {t} ({i})" for y, i, t in mid_a.itertuples(index=False)]
    report += ["", f"## Off-topic titles removed: {len(offtopic_rows)}", "",
               f"Dated before {EDITING_FROM}, found only by the CRISPR search, title has no biology word. "
               f"Check none of these is a real CRISPR paper.", ""]
    report += [f"- {int(y)}: {t} ({i})" for y, i, t in offtopic_rows.sort_values("year").itertuples(index=False)]
    kept_hand = [i for i in KEEP if i in set(df["doc_id"])]
    report += ["", f"## Kept by hand despite a removal rule: {len(kept_hand)} of {len(KEEP)} listed", ""]
    report += [f"- {i}: {KEEP[i]}" for i in KEEP] + (["", f"Not found in corpus: {sorted(set(KEEP) - set(kept_hand))}"]
                                                        if len(kept_hand) < len(KEEP) else [])
    if len(generic_survivors):
        report += ["", "### antecedent_generic papers kept (check these)", ""]
        report += [f"- {int(y) if pd.notna(y) else '?'}: {t}" for y, t in generic_survivors.sort_values("year").itertuples(index=False)]
    report += ["", "## Seeds not found", ""] + ([f"- {m}" for m in missing_seeds] or ["- none"])
    report += ["", "## Queries used", "", "```json", json.dumps(QUERIES, indent=2), "```"]
    report += ["", "## Filters used", "", "```json", json.dumps(FILTERS, indent=2), "```"]
    rp = os.path.join(outdir, "corpus_counts.md")
    open(rp, "w").write("\n".join(report))
    print("\n".join(report[:12 + len(t)]))
    print(f"\nWrote {path} and {rp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    if not API_KEY:
        sys.exit("OpenAlex needs an API key (free): openalex.org/settings/api, then set OPENALEX_API_KEY.")
    dry_run() if a.dry_run else full_pull(a.out)
