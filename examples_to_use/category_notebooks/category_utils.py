from pathlib import Path
import pandas as pd


# Common column alias map used across notebooks
COLUMN_ALIASES = {
    'Date': ['Date', 'DATE', 'Timestamp', 'Trading Date'],
    'Open': ['Open', 'OPEN', 'Open Price', 'OPEN_PRICE'],
    'High': ['High', 'HIGH', 'High Price', 'HIGH_PRICE'],
    'Low': ['Low', 'LOW', 'Low Price', 'LOW_PRICE'],
    'Close': ['Close', 'CLOSE', 'Close Price', 'CLOSE_PRICE', 'Prev Close'],
    'Volume': ['Volume', 'VOLUME', 'Shares Traded', 'Total Traded Quantity', 'TOTTRDQTY', 'Total Traded Volume'],
}


def _find_column(columns, candidates):
    normalized = {c.strip().lower(): c for c in columns}
    for candidate in candidates:
        match = normalized.get(candidate.strip().lower())
        if match:
            return match
    raise KeyError(f'Missing required column. Expected one of: {candidates}')


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(',', '', regex=False).str.replace('%', '', regex=False).str.strip()
    return pd.to_numeric(cleaned, errors='coerce')


def _parse_dates(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    for fmt in ('%d-%b-%Y', '%d-%b-%y', '%d %b %Y', '%Y-%m-%d'):
        parsed = pd.to_datetime(values, format=fmt, errors='coerce')
        if parsed.notna().any():
            return parsed
    return pd.to_datetime(values, errors='coerce')


def load_nse_history(csv_path):
    """Load an NSE-style OHLCV CSV and return normalized DataFrame.

    This function strips header whitespace, coerces numeric columns,
    parses common date formats and drops incomplete rows. It is
    intended as a shared helper for the example notebooks.
    """
    raw = pd.read_csv(csv_path)
    # Normalize header whitespace which caused duplicated alias logic
    raw.rename(columns=lambda c: c.strip() if isinstance(c, str) else c, inplace=True)

    normalized = pd.DataFrame()
    normalized['Date'] = _parse_dates(raw[_find_column(raw.columns, COLUMN_ALIASES['Date'])])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        source_col = _find_column(raw.columns, COLUMN_ALIASES[col])
        normalized[col] = _to_numeric(raw[source_col])

    normalized = normalized.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return normalized.sort_values('Date').reset_index(drop=True)


def prepare_category(csv_path, category_prefix, max_indicators=None):
    """Load CSV, compute TA features and return (df, df_ta, category_columns).

    Args:
        csv_path: Path-like to the NSE OHLCV CSV.
        category_prefix: prefix string to filter ta columns (e.g. 'volume_').
        max_indicators: optional int to truncate the returned columns list.

    Returns:
        (df, df_ta, category_columns)
    """
    # Ensure the project 'ta' package is importable when notebooks run
    import sys
    from pathlib import Path as _Path
    pkg_root = _Path(__file__).resolve().parents[2]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from ta import add_all_ta_features
    from ta.utils import dropna

    df = load_nse_history(csv_path)
    df = dropna(df)
    df_ta = add_all_ta_features(df.copy(), open='Open', high='High', low='Low', close='Close', volume='Volume', fillna=False)

    category_columns = [c for c in df_ta.columns if c.startswith(category_prefix)]
    if max_indicators is not None:
        category_columns = category_columns[:max_indicators]

    return df, df_ta, category_columns
