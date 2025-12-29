# API reference

This document describes the public API of the `fairness` package.

Conventions:
- Arrays may be Python lists, NumPy arrays, or pandas Series unless stated otherwise.
- Functions that operate on predictions expect `y_pred` as 0/1 labels (not probabilities), unless stated otherwise.
- All functions aim to preserve row order / index alignment where relevant.

---

## `fairness.data`

### `load_heart_csv(path: str | Path) -> pd.DataFrame`

Load the UCI Heart Disease dataset from a CSV file and return a cleaned DataFrame.

**Parameters**
- `path`: Path to CSV file.

**Returns**
- `pd.DataFrame`: DataFrame with standardised column names and expected dtypes.

**Notes**
- Intended as an adapter for the demo dataset. For general datasets, load data using pandas and follow the same schema.

**Example**
```python
from fairness.data import load_heart_csv
df = load_heart_csv("data/heart.csv")
```

---

## `fairness.preprocess`

### `add_age_group(df: pd.DataFrame, age_col: str = "Age", new_col: str = "age_group", bins: Sequence[float] = (0, 55, 120), labels: Sequence[str] = ("young", "older"),) -> pd.DataFrame:`

Add a categorical age-group column derived from a continuous age column.

**Parameters**
- `df`: Input dataset.
- `age_col`: Name of the column containing numeric ages.
- `new_col`: Name of the derived categorical column to create.
- `bins`: Bin edges passed to pandas.cut.
- `labels`: Labels assigned to the bins.

**Returns**
- `pd.DataFrame`: Copy of df with the new categorical column added.

**Raises**
- `ValueError`: If age_col is missing or binning produces missing values.

**Example**

```python
from fairness.preprocess import add_age_group
df = add_age_group(df_raw, age_col="Age", new_col="age_group", bins=(0, 55, 120), labels=("young", "older"))
```

