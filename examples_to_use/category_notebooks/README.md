# NIFTY 50 Category Notebooks (5 Files)

This folder provides a category-wise walkthrough of all 43 indicators listed in the project README.

## Notebook Index

- [01_volume_indicators.ipynb](01_volume_indicators.ipynb)
- [02_volatility_indicators.ipynb](02_volatility_indicators.ipynb)
- [03_trend_indicators.ipynb](03_trend_indicators.ipynb)
- [04_momentum_indicators.ipynb](04_momentum_indicators.ipynb)
- [05_others_indicators.ipynb](05_others_indicators.ipynb)

## Data Source

The sample dataset in this folder was downloaded from the official NSE historical index data page:

- https://www.nseindia.com/reports-indices-historical-index-data

Default dataset file used by notebooks:

- `NIFTY 50-02-04-2025-to-02-04-2026.csv`

## Notebook Behavior

Each notebook:

- reads NSE-style OHLCV CSV input
- computes TA features using `ta.add_all_ta_features`
- filters indicators for one category prefix
- renders interactive Plotly charts in final cells

No PNG files are written. Visual outputs remain embedded in notebook execution outputs.

## One-Go Indicator Generation (Single Script)

Use this script if you want one complete feature table directly, without running all notebooks.

```python
from pathlib import Path
import pandas as pd
from ta import add_all_ta_features

csv_path = Path("examples_to_use/category_notebooks/NIFTY 50-02-04-2025-to-02-04-2026.csv")
df = pd.read_csv(csv_path)

# Normalize common NSE column names.
df = df.rename(columns={
  "Date ": "Date",
  "Open ": "Open",
  "High ": "High",
  "Low ": "Low",
  "Close ": "Close",
  "Shares Traded ": "Volume",
})

for col in ["Open", "High", "Low", "Close", "Volume"]:
  df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"]).sort_values("Date")

df_all = add_all_ta_features(
  df=df,
  open="Open",
  high="High",
  low="Low",
  close="Close",
  volume="Volume",
  fillna=False,
)

print(f"Rows: {len(df_all)}")
print(f"Total columns: {len(df_all.columns)}")
print(df_all.columns)
```
