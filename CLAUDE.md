# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
python scripts/build_atc_mapping.py        # -> data/ingredient_to_atc4.parquet

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
python scripts/sync_defacto_to_catalog.py  # adds de facto combos (cohort_9*) into drug_catalog.jsonl

# 4. Biomarkers (LDL, HbA1c, eGFR, ALT, CRP, SBP)
python scripts/cache_biomarkers.py         # scans STARR measurement shards -> cache_full/biomarkers_measurements.parquet
python scripts/extract_biomarkers.py [--min-n 1]

# 5. Candidate comparator pairs -> data/candidate_pairs.parquet
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

  Rows are tagged by `record_type` (0 = cohort, 1 = attrition counts, 2 = de facto rows). De facto pairs build up in memory across the batch and are exported at the end.
- **`DrugCohort`** (`src/cohorts/cohort.py`) is saved to `<output_cohorts_dir>/cohort_<drug_id>/`. The folder holds `metadata.json` and `stanford_index.parquet` (person_id, t0, t_6m, t_12m, follow-up flags), plus `stanford_labs.parquet`, `biomarkers.parquet` and optional `.npy` tensors (MOTOR z0 768-d, RABIT deltas, UKB Olink, k-means prototypes). `data/cohorts/manifest.parquet` records one row per drug (`status` SAVED/ZERO_PATIENT, `n_final_stanford`). Downstream steps filter on `status == "SAVED" & n_final_stanford >= min_n`.
- **De facto combination IDs** are synthetic: `md5("{min}_{max}")[:12] % 1e12 + 9_000_000_000_000`. The same formula appears in `extractor.py` and `catalog.py` and must stay in sync. Their folders match `cohort_9*`, which is how `register_de_facto_cohorts` / `sync_defacto_to_catalog.py` find them. For a combination, `u = u_A + u_B`, and the text embedding is the normalized mean of the two.
- **`src/pairs/`**: `CandidatePair` has a permutation-invariant key `(min_id, max_id, stratum_concept_id)`. `PairRegistry` merges method flags on that key (`by_indication`, `by_css`, `by_kmeans`, `by_atc4`), and consensus means at least two methods agree. Only Method 1 (shared diagnoses at t0, `indication_pairs.py`) and intra-ATC4 pairs are implemented.

## Gotchas

- `utils/` is **legacy**: it imports `src.cohort_extractor`, which no longer exists. The maintained equivalents are in `scripts/` and `src/`. `utils/extract_motor_representations.py` (MOTOR/FEMR GPU extraction) has not been ported yet.
- `extract_cohorts.py` in resume mode skips drugs whose folders already exist, so de facto pairs are only collected from drugs processed in that run. Use `--no-resume` for a complete de facto set.
- `CohortExtractor` checks the cache schema on load. A "Cache obsolète" error means the cache was built by an older `build_cache.py` and must be regenerated.
- `codebase.txt` is a concatenated snapshot of the source files and may be out of date. Edit the real files, not this snapshot.
- `data/` and all parquet/npy/pt artifacts are gitignored (clinical data). Never commit them.
