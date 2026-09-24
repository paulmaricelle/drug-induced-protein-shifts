# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working rules (from the project owner — follow strictly)

- **Role: coding agent.** Goal is to move the project forward fast: edit code, run pipelines, debug. Act freely on technical work and on minor conceptual calls needed to fix a bug.
- **Research direction is the owner's call.** Project direction, the next major step, and architectural choices must be decided by the owner. When a technical step is done, report back and decide the next direction together. Proposing concepts (drawing on codebase knowledge) is welcome, but consult as soon as a conceptual question arises — don't let that slow down technical action.
- **No direct access to EHR data.** Never read, open, `head`, sample or print the contents of EHR folders (raw STARR OMOP, `data/cache_*`, cohort parquet files, labs, etc.) or any personal medical data. Writing code that reads them inside extraction pipelines is fine. Only aggregate statistics (cohort sizes under given conditions, attrition counts, pair counts) may be printed and inspected.

## Project overview

Research pipeline for **Target Trial Emulation (TTE)** of drug comparisons on Stanford STARR OMOP EHR data (UK Biobank Olink support is scaffolded in `DrugCohort` but not yet populated). It builds a drug catalog with biological target vectors, extracts new-user drug cohorts, attaches lab biomarkers, and generates candidate active-comparator pairs. Code comments, docstrings, CLI help and console output are in **French**, and docstrings cite protocol sections (e.g. "Section 4.2"). Keep that convention.

## Environment & commands

- Python 3.12 venv at `.venv/` (`source .venv/bin/activate`). There is no requirements file, test suite, linter config or build system. Validation is done through the `scripts/audit_*.py` scripts and `test-notebook.ipynb`.
- Run scripts **from the repo root**. Scripts add the repo root to `sys.path` themselves and import `src.*`. A few (`scripts/inspect_chembl.py`, `geometry_audit/*`) use CWD-relative `data/` paths.
- Indentation is mixed: some files use 2 spaces (`src/config.py`, `src/cohorts/extractor.py`, `scripts/extract_cohorts.py`, `scripts/audit_cohorts.py`, `scripts/cache_biomarkers.py`) and others use 4. Match the file you are editing.

Pipeline, in dependency order:

```bash
# 0. Vocabulary mappings (from /remote/shared/collab/omop-vocabularies/v20250227)
python scripts/build_mapping.py            # -> data/ingredient_to_prescriptions.parquet
python scripts/build_atc_mapping.py        # -> data/ingredient_to_atc4.parquet (vote majoritaire STARR si le cache existe + corrections src/catalog/atc4_overrides.csv)

# 1. Drug catalog: each flag is one stage and they are run in this order
python scripts/build_catalog.py --fetch-fastas | --embed-esm2 | --embed-reactome | --embed-string \
    | --embed-gtex | --assemble-vp | --build-catalog | --embed-text | --build-final-catalog
#    (--check-chembl only inspects the RxNorm<->ChEMBL mapping)
python scripts/audit_catalog.py

# 2. Local OMOP cache (default --ratio 0.2 -> data/cache_benchmark; 1.0 -> data/cache_full)
python scripts/build_cache.py --ratio 1.0

# 3. Cohorts. Defaults to the SAMPLE cache; --full is required for production
python scripts/extract_cohorts.py --full [--limit N] [--no-resume]
python scripts/audit_cohorts.py [--drugs ID ...]
python scripts/sync_defacto_to_catalog.py  # replaces the catalog's `combination` items with those on disk (cohort_9*)

# 4. Biomarkers (LDL, HbA1c, eGFR, ALT, CRP, SBP)
python scripts/cache_biomarkers.py         # scans STARR measurement shards -> cache_full/biomarkers_measurements.parquet
python scripts/extract_biomarkers.py [--min-n 1]

# 5. Candidate comparator pairs -> data/candidate_pairs.parquet (flags is_add_on / same_ingredients recomputed on every run)
python scripts/build_candidate_pairs.py --intra-atc4 --method1-indications [--min-n 100]
python scripts/build_candidate_pairs.py    # no flags = audit the registry
```

To exercise a single drug, run `scripts/extract_cohorts.py --limit N`, run `audit_cohorts.py --drugs <omop_ingredient_id>`, or instantiate `CohortExtractor(...).extract_cohort(drug_id)` in the notebook.

## Architecture

- **`src/config.py`**: `ProtocolConfig` holds the TTE parameters: 365-day pre-observation, 182-day minimum follow-up, 12-month horizon, 365-day washout on the drug and its ATC4 class, and a minimum of 50 patients for de facto combinations. `PathConfig` holds all paths. `is_sample` switches the cache (`data/cache_benchmark` vs `data/cache_full`) and the cohort output dir (`data/cohorts_benchmark` vs `data/cohorts`). The raw STARR OMOP path (`/remote/private/starr_omop_deid/...`) is PHI; only read it through the cache builders.
- **`src/catalog/`**: builds the `DrugCatalog` (`data/catalog/drug_catalog.jsonl`).
  - Each `DrugItem` is keyed by OMOP RxNorm ingredient `concept_id`. `kind` is one of `monotherapy`, `fixed_combination` or `de_facto_combination`.
  - Each item carries an ATC4 code, the descendant prescription concept IDs, a target vector `u_a` (1590-d) and an OMOP text embedding (1024-d).
  - `u_a` combines ChEMBL targets (`chembl.py`, `drug_features.py`: potency and agonist/antagonist direction) with per-protein features `v_p` (`protein_features.py`). `v_p` is the L2-normalized concatenation of ESM-2 650M, Reactome SVD (128), STRING (128) and GTEx (54).
  - `DrugCatalog` keeps inverse indexes: prescribed concept → drug in O(1), ATC4 → family, and ingredient pair → combination.
- **`src/cohorts/`**: `CohortExtractor` loads the parquet cache into in-memory DuckDB (16 threads, 48 GB). It runs one large SQL query per drug with these steps:
  - Take the first monotherapy exposure as t0.
  - Apply the observation-window filters.
  - Apply the washout against the target and its ATC4 comparators.
  - Keep "pure mono" patients: exactly one *new* ingredient at t0, where chronic renewals are allowed.
  - Collect de facto co-initiations: exactly two new ingredients on the same day.

  Rows are tagged by `record_type` (0 = cohort, 1 = attrition counts, 2 = de facto rows). De facto rows and fixed-product passes build up in memory across the batch and are merged and exported at the end (`export_combination_cohorts`).
- **`DrugCohort`** (`src/cohorts/cohort.py`) is saved to `<output_cohorts_dir>/cohort_<drug_id>/`. The folder holds `metadata.json` and `stanford_index.parquet` (person_id, t0, t_6m, t_12m, follow-up flags), plus `stanford_labs.parquet` (biomarkers written by `extract_biomarkers.py`; older cohorts used `biomarkers.parquet`) and optional `.npy` tensors (MOTOR z0 768-d, RABIT deltas, UKB Olink, k-means prototypes). `data/cohorts/manifest.parquet` records one row per drug (`status` SAVED/ZERO_PATIENT, `n_final_stanford`). Downstream steps filter on `status == "SAVED" & n_final_stanford >= min_n`.
- **Combination kinds and IDs.** IDs are synthetic 13-digit values computed only by `combo_drug_id()` in `src/catalog/catalog.py`.
  - `fixed_combination` (`8…`) is only an extraction *recipe* (the descendant codes of the combined product). It has no cohort folder and no manifest row, but stays in the catalog so extraction can be re-run.
  - `combination` (`9…`, folders `cohort_9*`, scanned by `register_combination_cohorts`) is the **single cohort per ingredient pair**, merging fixed-product and de facto patients. It inherits the fixed product's descendant codes, so prescribed-code lookup resolves to it.
  - The pair index is keyed by `(kind, pair)` (`DrugCatalog.get_combination`). Build combination items only with `DrugCatalog.build_combination_item`, which sets `u = u_A + u_B` and uses the normalized mean of the two text embeddings.
- **What each cohort contains:**
  - A *monotherapy* cohort has exactly one new ingredient at t0, taken as monotherapy.
  - A *combination* cohort is the union of two sources, recorded per patient in the `stanford_index` column `combo_source`:
    - `fixed`: t0 is the first dispensing of the combined product, and its two ingredients are the only new ones at t0;
    - `de_facto`: exactly two new monotherapies at t0, and the patient must be captured by **both** monotherapy passes, so the washout covers both ATC4 classes.
  - A patient present in both sources keeps the earliest t0. The manifest has `n_fixed` and `n_de_facto` columns. `min_de_facto_size` (50) applies only to pairs without a fixed product.
  - The washout of a combination covers the ATC4 families of both ingredients (`DrugCatalog.get_comparator_ids`).
- **`src/pairs/`**: `CandidatePair` has a permutation-invariant key `(min_id, max_id, stratum_concept_id)`. `PairRegistry` merges method flags on that key (`by_indication`, `by_css`, `by_kmeans`, `by_atc4`), and consensus means at least two methods agree. Only Method 1 (shared diagnoses at t0, `indication_pairs.py`) and intra-ATC4 pairs are implemented.

## Gotchas

- `utils/` is **legacy**: it imports `src.cohort_extractor`, which no longer exists. The maintained equivalents are in `scripts/` and `src/`. `utils/extract_motor_representations.py` (MOTOR/FEMR GPU extraction) has not been ported yet.
- `extract_cohorts.py` in resume mode skips drugs whose folders already exist, so de facto rows are only collected from drugs processed in that run. Use `--no-resume` for a complete set. A full `--no-resume` run (no `--limit`) deletes cohort folders absent from the new manifest.
- `CohortExtractor` checks the cache schema on load. A "Cache obsolète" error means the cache was built by an older `build_cache.py` and must be regenerated.
- ATC4 choice: `build_atc_mapping.py` weights each candidate ATC4 by STARR exposures to the mono-ingredient forms linked to it, so it needs the cache. `src/catalog/atc4_overrides.csv` holds manual corrections (versioned, with a justification per row). They override the vote both in the mapping and in `DrugCatalog.from_pipeline_artifacts`. Add rows there rather than patching generated files.
- `codebase.txt` is a concatenated snapshot of the source files and may be out of date. Edit the real files, not this snapshot.
- `PairRegistry` merges into the existing `data/candidate_pairs.parquet`. Delete that file before regenerating pairs after a re-extraction, or pairs with stale IDs will survive.
- `data/` and all parquet/npy/pt artifacts are gitignored (clinical data). Never commit them.
