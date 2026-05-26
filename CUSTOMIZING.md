# Customizing Object Names

The demo uses these default names, which `setup.sql` creates and `src/config.py`
references. A fresh clone + `setup.sql` run requires no edits.

| Object | Default name |
|--------|-------------|
| Database | `ACTUARIAL_DEMO_DB` |
| Schema | `ACTUARIAL_PRICING` |
| Warehouse | `ACTUARIAL_DEMO_WH` |
| Compute pool | `ACTUARIAL_DEMO_POOL` |
| Role | `ACTUARIAL_DEMO_ROLE` |

## To rename objects

Update **all three locations** — they must match:

### 1. `setup.sql` — change the `SET` variables at the top
```sql
SET db_name    = 'YOUR_DATABASE';
SET schema_name = 'YOUR_SCHEMA';
SET wh_name    = 'YOUR_WAREHOUSE';
SET pool_name  = 'YOUR_COMPUTE_POOL';
SET role_name  = 'YOUR_ROLE';
```

### 2. `create-table.sql` — change the matching `SET` variables at the top
```sql
SET db_name    = 'YOUR_DATABASE';
SET schema_name = 'YOUR_SCHEMA';
SET wh_name    = 'YOUR_WAREHOUSE';
SET role_name  = 'YOUR_ROLE';
```

### 3. `src/config.py` — change the Python constants to match
```python
DATABASE     = "YOUR_DATABASE"
SCHEMA       = "YOUR_SCHEMA"
WAREHOUSE    = "YOUR_WAREHOUSE"
COMPUTE_POOL = "YOUR_COMPUTE_POOL"
ROLE         = "YOUR_ROLE"
```

**That's it.** The notebook imports from `src/config.py`, and the training scripts
(`train.py`, `train_glm.py`, `train_gbm_distributed.py`) import from `config.py`
when running as ML Jobs — no other files need editing.

### 4. SQL notebooks and standalone scripts

These files contain Snowflake object names directly (they cannot import from `config.py`)
and need updating if you rename objects:

- `notebooks/data_dictionary_guide.ipynb` — `ACTUARIAL_DEMO_DB` appears in SQL cells (~15 places); do a find-and-replace in the raw file
- `train-handson.py` — standalone training script; update the inline constants (lines 23, 73–76)

## Keeping customized values out of git

For private deployments, fork this repository and make your fork private before
editing `src/config.py`. This keeps customer-specific names out of the public repo.
