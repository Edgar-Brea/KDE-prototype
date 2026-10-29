# The CRISPR corpus

## Pull

Papers were pulled from [OpenAlex](https://openalex.org) on 8 October 2026 with `corpus_pull.py` (in this folder, for provenance; it needs an `OPENALEX_API_KEY`). Each search covers titles and abstracts; work types article, review and preprint.

| Search (`strand`) | Query | Years |
|---|---|---|
| `crispr` | "CRISPR" OR "clustered regularly interspaced short palindromic repeats" | all |
| `antecedent_named` | "short regularly spaced repeats" OR "regularly spaced repeats" OR "spacers interspersed direct repeats" OR "large clusters of tandem repeats" | to 2011 |
| `antecedent_generic` | ("direct repeats" OR "tandem repeats" OR "repetitive sequences") AND (archaea OR archaeal OR prokaryotic OR "bacterial genome") | to 2005 |
| `antecedent_phage` | "phage resistance" AND ("Streptococcus thermophilus" OR "lactic acid bacteria") | to 2007 |
| `antecedent_spoligo` | "spoligotyping" OR "direct repeat locus" OR "DR locus" | to 2005 |
| `seed` | 9 landmark papers added by DOI (Ishino 1987 to Jinek 2012) | |

Cleaning, applied to the downloaded records:
- **Relevance filters:** a paper is kept only if its title or abstract matches its search's vocabulary (removed 10,541 from `crispr`, 171 from `antecedent_generic`, 324 from `antecedent_spoligo`).
- **Record errors:** 523 papers dated before 2002 that mention CRISPR, or before 2012 that use genome-editing terms, and 17 early papers with off-topic titles, are OpenAlex records carrying the wrong abstract, and are removed.
- **Duplicates:** 4,584 removed (same DOI, or same title and year).

Result: **82,366 papers** (`crispr` strand 81,980; a paper can be found by several searches).

## Exclusions

| | Papers |
|---|---|
| Corpus | 82,366 |
| No abstract | 10,017 |
| Abstract under 200 characters | 835 |
| **Coded** | **71,514** |
| …of which over 5,000 characters (probably body text: `status = excluded_long_text`) | 1,592 |
| …retained (`status = retained`) | 69,922 |

Long texts stay in the exports with their status, so anyone can re-filter. Figure 1 uses retained abstracts from the `crispr` strand, 2002 on: 69,571 abstracts and 836,857 components.

## Extraction run

- **Model:** DeepSeek V4.1 Flash (`deepseek-flash`), thinking disabled, temperature 0, JSON output; frozen prompt `system_v4.md` and codebook `codebook_v3.md`.
- **Output:** 879,376 components from 71,514 abstracts (12.3 per abstract); no API errors.
- **Tokens:** 229.9 M input (83% cache hits), 53.3 M output.
- **Cost:** USD 38.43 (estimated from token counts at DeepSeek's off-peak rates; USD 0.54 per 1,000 abstracts).
- **Run time:** about 10.7 hours of processing (8 to 16 parallel requests), off-peak hours only.
