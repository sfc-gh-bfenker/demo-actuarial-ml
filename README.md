# Homeowners Insurance Precision Pricing
## Actuarial Loss Modeling with Snowflake

End-to-end actuarial pricing demo built on a ~678K policy homeowners portfolio.
Covers feature engineering, GBM training, experiment tracking, model registration,
batch scoring, and SHAP explainability — all running on Snowflake compute.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Snowflake account | With SPCS enabled for ML Jobs and batch inference |
| Compute pool | For ML Job training and `run_batch` scoring |
| Virtual warehouse | For SQL compute and Snowpark operations |
| Snowflake Notebooks | For running the interactive demo |
| Python ≥ 3.11 | Only needed for local setup (`load_actuarial_data.py`) |

---

## Setup

### 1. Run setup.sql

Open `setup.sql` in a Snowflake worksheet and run it top-to-bottom.
Requires **ACCOUNTADMIN**. Creates the database, schema, warehouse, compute pool,
role, and stages. No editing needed — defaults match what the demo expects.

> To use custom object names, see [CUSTOMIZING.md](CUSTOMIZING.md).

### 2. Load data

```bash
python src/load_actuarial_data.py
```

Downloads the freMTPL2 dataset from [OpenML](https://www.openml.org/d/41214),
converts it to XML, uploads it to the Snowflake stage, and creates the
`HOME_POLICY_FREQ` and `HOME_POLICY_SEV` tables (~2 min).
Pass `--help` for authentication options.

### 3. Create tables (optional)

Run `create-table.sql` in a Snowflake worksheet to re-create the tables from
the staged XML. This is only needed if you want to reload without re-running the
Python script — `load_actuarial_data.py` already creates both tables.

### 4. Run the notebook

Open `notebooks/actuarial_pricing_demo.ipynb` in Snowflake Notebooks or a local
Jupyter environment.

---

## Running the Demo Notebook

Upload `actuarial_pricing_demo.ipynb` to Snowflake Notebooks and open it.
The notebook runs entirely on Snowflake compute — no local Python environment needed.

### Section 1 — Configuration

The second cell imports from `config.py`. If you are running from a local IDE,
ensure `config.py` is in the same directory. In Snowflake Notebooks you can
also set the variables directly in that cell.

### Sections 1–4 — Data loading and feature engineering

Two implementations of the same feature engineering pipeline are shown:

- **Snowpark Connect (PySpark API)** — zero migration effort for teams with existing PySpark code
- **Native Snowpark (recommended)** — full Snowflake Horizon lineage; run this one if you are
  starting fresh

Both write their output to `ML_INPUT_SNOWFLAKE`. Run whichever matches your team's background,
or run both to compare. The rest of the notebook reads from `ML_INPUT_SNOWFLAKE`.

### Section 5 — EDA

Exposure-weighted histograms of the key rating factors. A good place to add your
own segmentation plots — try slicing by `TERRITORY_CODE` or `CONSTRUCTION_TYPE`
to see geographic and structural risk concentration.

### Section 6 — Feature Store

Creates a versioned feature view and generates training/validation datasets.

**Version numbers to watch:**

```python
fv = fs.register_feature_view(..., version="1", ...)   # bump when feature engineering changes
training_dataset = fs.generate_dataset(..., version="2", ...)  # bump for a new train/val split
```

Increment these when you re-run feature engineering or want a fresh split.
The downstream training cells reference the same version strings — keep them in sync.

### Section 7 — GBM Training

Trains a Snowflake-managed `XGBRegressor` directly on the Feature Store datasets.

> **In a Snowflake Notebook**, the Snowpark ML stored procedure that runs training
> doesn't inherit the notebook session's schema context. Call `.cache_result()` on
> the training DataFrames before passing them to `fit()` to avoid a
> `'ACTUARIAL_TRAINING' does not exist` error:
> ```python
> train_ds_sdf = training_dataset.read.to_snowpark_dataframe().cache_result()
> val_ds_sdf   = validation_dataset.read.to_snowpark_dataframe().cache_result()
> ```

### Sections 8–9 — Diagnostic charts

Double-lift chart and Lorenz / Gini curve — standard actuarial filing exhibits.
These are also uploaded as artifacts to the Snowflake Experiment run automatically.

### Section 10 — Model Registry

Registers the model in the Snowflake Model Registry. The version name (`"V2"`) and
model name (`"HOMEOWNERS_PURE_PREMIUM_GBM"`) can be changed freely — they're just labels.

### Section 11 — Remote training with ML Jobs

Submits `train.py` as an async job to the compute pool. This is the production path:
training runs on Snowflake compute, the model is registered automatically, and the
job name becomes the model version so every run is uniquely traceable.

Make sure `payload_stage` exists in your schema before submitting:

```sql
CREATE STAGE IF NOT EXISTS payload_stage
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

The dataset version passed via `--ds-version` should match whatever version you
generated in Section 6.

### Section 12 — Batch inference

Scores the full validation portfolio using `mv.run_batch()` on the compute pool.
The model name in `registry.get_model("ACTUARIAL_GBM")` should match what was
registered by `train.py` — update it if you changed `model_name` in `train.py`.

### Section 13 — SHAP explainability

Generates per-policy SHAP values using the Model Registry's built-in `explain`
function. This works because `train.py` registers models with
`options={"enable_explainability": True}` — no re-training needed.

---

## Key Files

| File | Purpose |
|---|---|
| `config.py` | Single place to configure database, schema, role, warehouse, compute pool |
| `actuarial_pricing_demo.ipynb` | Interactive demo notebook |
| `train.py` | Standalone training script for ML Jobs (imports from `config.py`) |
| `create-table.sql` | Pure-SQL data loading from XML stage |
| `load_actuarial_data.py` | Downloads freMTPL2 and uploads as XML (run locally) |
| `helpers.py` | `lorenz_curve` and `double_lift_chart` plotting utilities |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Dataset ACTUARIAL_TRAINING does not exist` | Feature Store datasets haven't been generated yet | Run the notebook through Section 6 (Feature Store) before submitting `train.py` |
| `'ACTUARIAL_TRAINING' does not exist` (during `fit()`) | Snowpark ML stored proc doesn't inherit notebook schema context | Call `.cache_result()` on training DataFrames before `fit()` |
| `Stage 'PAYLOAD_STAGE' does not exist or not authorized` | Stage owned by ACCOUNTADMIN, demo role lacks access | Transfer ownership: `GRANT OWNERSHIP ON STAGE PAYLOAD_STAGE TO ROLE ACTUARIAL_DEMO_ROLE COPY CURRENT GRANTS` |
| `Stages cannot currently be created in a personal database` | `submit_file` trying to create the payload stage in a personal DB | Pre-create the stage in your schema; use a fully-qualified `stage_name` |
| `USE WAREHOUSE` rejected in ML Job | `USE WAREHOUSE` is blocked inside SPCS containers | Pass `query_warehouse=WAREHOUSE` to `submit_file`; do not call `use_warehouse()` in the script |
| `Object does not exist` on `USE WAREHOUSE` | Job session role doesn't have USAGE on the warehouse | `GRANT USAGE ON WAREHOUSE <wh> TO ROLE <role>` |

