"""Figure 1: type composition over time (100% stacked area, streamgraph style).

  python -m src.analysis.fig1_type_composition [--config config.yaml] [--out figures]

Reads only exports/ (no corpus, no API key), retained abstracts with a year only, restricted to papers whose
strand includes analysis.fig1_strand and published from analysis.fig1_from_year on. Each band is one type's
share of all components in that year. Smoothing pools component counts over a centred window of years and
then takes shares; band edges are a shape-preserving (PCHIP) spline between years, renormalised to 100%.
Years with no abstracts are left empty; years with fewer than analysis.thin_year_abstracts are faded.

Writes fig1_type_composition.png (300 dpi) and .pdf,
fig1_data.csv and fig1_caption.md (regenerated on every run).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

from src.analysis.common import RETAINED, TYPES, load_config, read_exports  # noqa: E402

INK, MUTED, GRID, FADE_BG = "#1f1f1f", "#6b6b6b", "#d9d9d9", "#ececec"
LIGHT_FILLS = ("phenomenon", "principle")    # dark text on these, white on the rest


def year_table(comps, index, window):
    """Per calendar year: counts per type, shares, smoothed shares (pooled counts), n_abstracts."""
    years = np.arange(int(index.year.min()), int(index.year.max()) + 1)
    counts = comps.groupby(["year", "type"]).size().unstack(fill_value=0).reindex(
        index=years, columns=list(TYPES), fill_value=0)
    n_abs = index.groupby("year").size().reindex(years, fill_value=0)
    share = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    pooled = counts.rolling(window, center=True, min_periods=1).sum()
    smooth = pooled.div(pooled.sum(axis=1).replace(0, np.nan), axis=0)
    smooth[n_abs == 0] = np.nan                  # no abstracts that year: no band
    return counts, share, smooth, n_abs


def runs(years):
    """Split sorted years into runs of consecutive years."""
    out, cur = [], [years[0]]
    for y in years[1:]:
        if y == cur[-1] + 1:
            cur.append(y)
        else:
            out.append(cur)
            cur = [y]
    return out + [cur]


def band_curves(shares, smooth):
    """Yield (x, stacked shares [n_types, len(x)]) per run of consecutive defined years."""
    shares = shares.dropna()
    for r in runs(list(shares.index)):
        s = shares.loc[r].to_numpy()
        if smooth and len(r) > 1:
            x = np.linspace(r[0] - 0.5, r[-1] + 0.5, 40 * len(r) + 1)
            xe = np.clip(x, r[0], r[-1])          # flat for the half year beyond each end
            v = np.vstack([PchipInterpolator(r, s[:, k])(xe) for k in range(s.shape[1])]).clip(min=0)
            v /= v.sum(axis=0)
        else:                                     # one block per year
            x = np.repeat(np.array(r, float), 2) + np.tile([-0.5, 0.5], len(r))
            v = np.repeat(s, 2, axis=0).T
        yield x, v


def spans(years):
    return [(r[0] - 0.5, r[-1] + 0.5) for r in runs(list(years))] if len(years) else []


def in_strand(df, strand):
    """Rows whose |-joined strand column includes strand (all rows if strand is None)."""
    if strand is None:
        return df
    return df[df.strand.fillna("").str.split("|").map(lambda s: strand in s)]


def prepare(comps, index, acfg):
    """Retained abstracts with a year, in scope (strand, start year) -> year table and thin years."""
    index = in_strand(index[(index.status == RETAINED) & index.year.notna()], acfg.get("fig1_strand"))
    comps = in_strand(comps[(comps.status == RETAINED) & comps.year.notna()], acfg.get("fig1_strand"))
    start = acfg.get("fig1_from_year")
    if start is not None:
        index, comps = index[index.year >= start], comps[comps.year >= start]
    counts, share, smooth, n_abs = year_table(comps, index, acfg["smoothing_years"])
    thin = n_abs.index[n_abs < acfg["thin_year_abstracts"]]
    return counts, share, smooth, n_abs, thin


def draw(shares, n_abs, thin, colours, smooth, path, tick_every=5, dpi=300):
    fig, (ax, sx) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [5, 1], "hspace": 0.06})
    lo, hi = n_abs.index[n_abs > 0].min() - 0.5, n_abs.index.max() + 0.5
    fade = spans(thin)
    for a, b in fade:
        ax.axvspan(a, b, color=FADE_BG, lw=0, zorder=0)
        sx.axvspan(a, b, color=FADE_BG, lw=0, zorder=0)
    labelled = None
    for x, v in band_curves(shares, smooth):
        top = np.cumsum(v, axis=0)
        bottom = np.vstack([np.zeros(len(x)), top[:-1]])
        for k, t in enumerate(TYPES):
            ax.fill_between(x, bottom[k], top[k], color=colours[t], lw=0, zorder=1)
        for k in range(len(TYPES) - 1):           # thin surface gap between bands
            ax.plot(x, top[k], color="white", lw=0.6, zorder=2)
        if labelled is None or len(x) > len(labelled[0]):
            labelled = (x, bottom, top)
    for a, b in fade:                             # fade thin years over the bands
        ax.axvspan(a, b, color="white", alpha=0.55, lw=0, zorder=3)

    # direct labels inside the longest run, where the bands are widest
    x, bottom, top = labelled
    at = int(np.argmin(np.abs(x - (x[0] + 0.8 * (x[-1] - x[0])))))
    for k, t in enumerate(TYPES):
        h = top[k][at] - bottom[k][at]
        if h > 0.05:
            ax.text(x[at], (bottom[k][at] + top[k][at]) / 2, t.capitalize(), ha="center", va="center",
                    fontsize=8.5, color=INK if t in LIGHT_FILLS else "white", zorder=4)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Share of components", color=INK, fontsize=9)

    yrs = n_abs.index[n_abs > 0]
    sx.bar(yrs, n_abs[yrs], width=0.8, color=MUTED, lw=0, zorder=1)
    sx.set_yscale("log")
    top_tick = 10 ** int(np.ceil(np.log10(n_abs.max())))
    ticks = [t for t in (1, 100, 10_000, 1_000_000) if t <= top_tick]
    sx.set_ylim(0.3, top_tick * 1.2)          # so a year with one abstract still shows a bar
    sx.set_yticks(ticks)
    sx.set_yticklabels([f"{t:,}" for t in ticks])
    sx.minorticks_off()
    sx.set_ylabel("Abstracts\n(log scale)", color=INK, fontsize=8)

    sx.set_xlim(lo, hi)
    first = int(np.ceil((lo + 0.5) / tick_every) * tick_every)
    sx.set_xticks(range(first, int(hi) + 1, tick_every))
    for a in (ax, sx):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=8, length=3)
    sx.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", help="output folder (default: analysis.figures_dir)")
    args = ap.parse_args()
    _, acfg = load_config(args.config)
    out = args.out or acfg["figures_dir"]
    os.makedirs(out, exist_ok=True)

    comps, index = read_exports(acfg)
    counts, share, smooth, n_abs, thin = prepare(comps, index, acfg)
    window, colours, thin_n = acfg["smoothing_years"], acfg["colours"], acfg["thin_year_abstracts"]
    tick_every, strand = acfg.get("fig1_tick_years", 5), acfg.get("fig1_strand")
    n_index, n_comps = int(n_abs.sum()), int(counts.to_numpy().sum())

    fig = draw(smooth, n_abs, thin, colours, True, os.path.join(out, "fig1_type_composition.png"), tick_every)
    fig.savefig(os.path.join(out, "fig1_type_composition.pdf"), bbox_inches="tight",
                metadata={"CreationDate": None})   # no timestamp, so reruns give identical files
    plt.close(fig)

    long = lambda df, name: df.rename_axis("year").reset_index().melt("year", var_name="type", value_name=name)
    data = long(counts, "count").merge(long(share, "share")).merge(long(smooth, "smoothed_share"))
    data = data.merge(n_abs.rename("n_abstracts"), left_on="year", right_index=True)
    data["type"] = pd.Categorical(data["type"], TYPES)
    data.sort_values(["year", "type"]).to_csv(os.path.join(out, "fig1_data.csv"), index=False,
                                              float_format="%.4f")

    present = n_abs.index[n_abs > 0]
    first = present.min()
    empty = [y for y in n_abs.index if n_abs[y] == 0]
    faded = [y for y in thin if n_abs[y] > 0]
    scope = (f"Only papers found by the `{strand}` search strand are shown" +
             (f": {acfg['fig1_scope_note'].strip()}" if acfg.get("fig1_scope_note") else ".") +
             " All strands stay coded and released with their `strand` column. ") if strand else ""
    cap = [
        "# Figure 1 caption", "",
        "_Generated by `python -m src.analysis.fig1_type_composition`; numbers update on every run._", "",
        "**Figure 1. Composition of knowledge components by type, "
        f"{first}–{present.max()}.** Each band is one type's share of all components coded in "
        "abstracts published that year; the bands always sum to 100%. From bottom to top: "
        f"{', '.join(TYPES)}. {scope}Shares are smoothed with a {window}-year centred "
        f"window: component counts are pooled across the {window} years and shares taken from the pooled counts "
        "(at the first and last year, and next to years with no abstracts, the window holds only the years that "
        "have them). Band edges are drawn with a shape-preserving spline between years. Years with fewer than "
        f"{thin_n} abstracts are faded" + ("; years with no abstracts are left blank" if empty else "") + ". "
        "The strip below shows the number of abstracts per year on a log scale. "
        f"n = {n_index:,} abstracts, {n_comps:,} components; abstracts over {acfg['long_text_chars']:,} "
        "characters (probably body text) and papers without a publication year are excluded.", "",
        f"Faded years ({len(faded)}): {', '.join(map(str, faded))}.", "",
        f"Years with no abstracts inside the range ({len(empty)}): {', '.join(map(str, empty)) or 'none'}.", "",
    ]
    open(os.path.join(out, "fig1_caption.md"), "w", encoding="utf-8").write("\n".join(cap))
    print(f"{n_index:,} abstracts, {n_comps:,} components, years {first}–{present.max()}")
    print(f"faded ({len(faded)}): {faded}")
    print(f"empty ({len(empty)}): {empty}")

if __name__ == "__main__":
    main()
