#!/usr/bin/env python3
"""
KDE · draw_samples.py — draw the disjoint coding sets from the corpus (provenance of the test set).

  python evaluation/draw_samples.py   # draws once, then refuses to redraw (sets are frozen)
  python evaluation/draw_samples.py --force  # redraw ONLY if nobody has started coding yet

Reads:  data/corpus.parquet   (made by results/corpus_pull.py)
Writes: data/samples/
          test_RA.csv          50 abstracts for the RA (sample_id, title, abstract) — shuffled, no metadata
          overlap_eb.csv       10 of those 50, for a second coder to code blind after the codebook is frozen
          pilot.csv            the author's schema pilot (coded by hand to revise the codebook)
          dev.csv              prompt development (model output reviewed by the author)
          practice.csv         RA training before the test set
          assignments.csv      every drawn paper: set, sample_id, doc_id, period, stratum, year, title
          samples_manifest.md  seed, sizes, allocation, eligibility counts — the record of the draw

Design (change the settings block, not the code):
  - Sets never overlap, except overlap ⊂ test by design.
  - The test set is drawn FIRST, so changing pilot/dev/practice sizes never changes it.
  - Each set is split across periods (ALLOCATION), oversampling the old periods: they are small,
    hardest to code, and the comparison over time needs them.
  - Within a period, papers are spread evenly across strata (search strand; CRISPR papers by
    5-year bin), so TB strain-typing papers can't fill the old-period quota on their own.
    No set takes more than MAX_SHARE of a stratum, so the small strata also reach the pilot.
  - Eligible = has a year, an English abstract of >= MIN_WORDS words, and is not a landmark seed
    (frontier models know those papers well, which would flatter the test).
  - Fixed SEED: same corpus + same settings = same sets, on any machine.
"""
import argparse, os, sys, json, time
import numpy as np
import pandas as pd

# ===================== SETTINGS =====================
SEED = 20261008
MIN_WORDS = 50

# papers per period, per set. Test is drawn first, then the others in this order.
ALLOCATION = {
    "test":     {"antecedent": 15, "early": 15, "post": 20},   # 50, coded by the RA
    "pilot":    {"antecedent": 3,  "early": 3,  "post": 4},    # 10, the author's schema pilot
    "dev":      {"antecedent": 5,  "early": 5,  "post": 5},    # 15, prompt development
    "practice": {"antecedent": 1,  "early": 2,  "post": 2},    # 5, RA training
}
OVERLAP = {"antecedent": 3, "early": 3, "post": 4}             # 10 of the test set, second coder, blind
MAX_SHARE = 0.6   # no set takes more than 60% of any stratum, so tiny strata reach the pilot too

PREFIX = {"test": "T", "pilot": "P", "dev": "D", "practice": "R"}
STRAND_ORDER = ["antecedent_named", "antecedent_phage", "antecedent_generic", "antecedent_spoligo"]
YEAR_BINS = [(2002, 2006), (2007, 2011), (2012, 2015), (2016, 2020), (2021, 9999)]
# ====================================================


def stratum(matched, year):
    tags = matched.split("|")
    if "crispr" in tags:
        for lo, hi in YEAR_BINS:
            if lo <= year <= hi:
                return f"crispr {lo}-{hi if hi < 9999 else 'now'}"
        return "crispr other"
    for s in STRAND_ORDER:
        if s in tags:
            return s.replace("antecedent_", "")
    return "other"


def spread_draw(pool, n, rng, sizes):
    """Take n papers spread evenly across strata: shuffle each stratum, then take one from each
    in turn (strata in a fixed order). Small strata are therefore over-represented, by design,
    but each stratum gives at most MAX_SHARE of its full size (sizes) unless that leaves n unmet."""
    groups = {k: list(rng.permutation(g.index.to_numpy())) for k, g in sorted(pool.groupby("stratum"))}
    cap = {k: max(1, int(MAX_SHARE * sizes[k])) for k in groups}   # sizes = this period's strata only
    used = {k: 0 for k in groups}
    picked = []
    for capped in (True, False):                     # second pass only if the caps leave n unmet
        while len(picked) < n:
            took = False
            for k in groups:
                if groups[k] and len(picked) < n and (not capped or used[k] < cap[k]):
                    picked.append(groups[k].pop(0)); used[k] += 1; took = True
            if not took:
                break
    if len(picked) < n:
        sys.exit(f"Not enough eligible papers: wanted {n}, pool has {len(pool)}.")
    return picked


def main(corpus, outdir, force):
    marker = os.path.join(outdir, "assignments.csv")
    if os.path.exists(marker) and not force:
        sys.exit(f"Samples already drawn ({marker}). They are frozen so coding stays valid.\n"
                 f"Only if NOBODY has started coding: rerun with --force.")

    df = pd.read_parquet(corpus) if corpus.endswith(".parquet") else pd.read_csv(corpus)
    n_all = len(df)
    words = df["abstract"].fillna("").str.split().str.len()
    elig = (df["period"].notna() & df["year"].notna() & (words >= MIN_WORDS)
            & (df["language"].fillna("en") == "en") & ~df["matched"].str.contains("seed"))
    reasons = {
        "no year / period": int((df["period"].isna() | df["year"].isna()).sum()),
        f"abstract missing or < {MIN_WORDS} words": int((words < MIN_WORDS).sum()),
        "not English": int((df["language"].fillna("en") != "en").sum()),
        "landmark seed": int(df["matched"].str.contains("seed").sum()),
    }
    pool = df[elig].copy()
    pool["stratum"] = [stratum(m, int(y)) for m, y in zip(pool["matched"], pool["year"])]

    rng = np.random.default_rng(SEED)
    taken, rows = set(), []
    for setname, alloc in ALLOCATION.items():
        ids = []
        for period, n in alloc.items():
            sizes = pool.loc[pool["period"] == period, "stratum"].value_counts().to_dict()
            avail = pool[(pool["period"] == period) & ~pool.index.isin(taken)]
            got = spread_draw(avail, n, rng, sizes)
            taken.update(got); ids += got
        order = rng.permutation(len(ids))                       # shuffle so periods aren't in blocks
        for k, j in enumerate(order, 1):
            r = pool.loc[ids[j]]
            rows.append(dict(set=setname, sample_id=f"{PREFIX[setname]}{k:02d}", doc_id=r["doc_id"],
                             period=r["period"], stratum=r["stratum"], year=int(r["year"]),
                             title=r["title"], abstract=r["abstract"]))
    a = pd.DataFrame(rows)

    # overlap: a period-balanced subset of the test set
    t = a[a["set"] == "test"]
    ov = []
    for period, n in OVERLAP.items():
        cand = t[t["period"] == period]
        ov += list(rng.choice(cand["sample_id"].to_numpy(), size=n, replace=False))
    a["eb_overlap"] = a["sample_id"].isin(ov) & (a["set"] == "test")

    assert a["doc_id"].is_unique, "sets overlap — this should never happen"

    os.makedirs(outdir, exist_ok=True)
    blind = ["sample_id", "title", "abstract"]                  # what coders see
    a[a["set"] == "test"].sort_values("sample_id")[blind].to_csv(os.path.join(outdir, "test_RA.csv"), index=False)
    a[a["eb_overlap"]].sort_values("sample_id")[blind].to_csv(os.path.join(outdir, "overlap_eb.csv"), index=False)
    for s in ["pilot", "dev", "practice"]:
        a[a["set"] == s].sort_values("sample_id")[blind].to_csv(os.path.join(outdir, f"{s}.csv"), index=False)
    a.drop(columns=["abstract"]).sort_values(["set", "sample_id"]).to_csv(marker, index=False)

    # manifest
    m = [f"# Sample draw — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
         f"Corpus: `{corpus}` · {n_all} papers · eligible: {len(pool)} · seed {SEED}", "",
         "## Not eligible (a paper can fail several)", "", "| reason | papers |", "|---|---|"]
    m += [f"| {k} | {v} |" for k, v in reasons.items()]
    m += ["", "## Eligible pool by stratum", "", "| period | stratum | papers |", "|---|---|---|"]
    for (p, s), n in pool.groupby(["period", "stratum"]).size().items():
        m.append(f"| {p} | {s} | {n} |")
    m += ["", "## Drawn", "", "| set | period | stratum | papers |", "|---|---|---|---|"]
    for (st, p, s), n in a.groupby(["set", "period", "stratum"]).size().items():
        m.append(f"| {st} | {p} | {s} | {n} |")
    m += ["", f"Second coder's blind overlap: {', '.join(sorted(ov))}", "",
          "## Settings", "", "```json", json.dumps(dict(SEED=SEED, MIN_WORDS=MIN_WORDS, ALLOCATION=ALLOCATION,
                                                         OVERLAP=OVERLAP, MAX_SHARE=MAX_SHARE, YEAR_BINS=YEAR_BINS), indent=2), "```"]
    open(os.path.join(outdir, "samples_manifest.md"), "w").write("\n".join(m))
    print("\n".join(m))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.parquet")
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.corpus, a.out, a.force)
