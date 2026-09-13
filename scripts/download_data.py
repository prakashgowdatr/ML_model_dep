"""
scripts/download_data.py — Download the UCI Beijing PM2.5 dataset.

The dataset (PRSA_Data_2010.1.1-2014.12.31.csv) contains ~43,000 hourly
rows with PM2.5, temperature, humidity, wind speed, and pressure.

Run:
    python scripts/download_data.py

The file is saved to:  data/raw/PRSA_data.csv
"""

import os
import sys
from pathlib import Path

# ── Make sure we can import from src/ ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CFG  # noqa: E402

# UCI ML Repository direct CSV link
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00381/PRSA_data_2010.1.1-2014.12.31.csv"
)

# Column rename map: raw UCI names → our feature names
COLUMN_MAP = {
    "pm2.5": "pm25",
    "TEMP": "temperature",
    "DEWP": "humidity",      # dew point as humidity proxy
    "Iws":  "wind_speed",    # cumulated wind speed (actual UCI column name)
    "PRES": "pressure",
}


def download() -> Path:
    """Download dataset and save to data/raw/PRSA_data.csv."""
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        print("❌  'requests' not installed. Run: pip install requests")
        sys.exit(1)

    out_path = PROJECT_ROOT / CFG.data.raw_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"✅  Dataset already exists at {out_path}. Skipping download.")
        return out_path

    print(f"⬇️   Downloading dataset from UCI repository …")
    try:
        resp = requests.get(DATASET_URL, timeout=30, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"\n❌  Download failed: {e}")
        print("\n📋  Manual download instructions:")
        print("    1. Open: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data")
        print("    2. Download 'PRSA_data_2010.1.1-2014.12.31.csv'")
        print(f"    3. Save it to: {out_path}")
        sys.exit(1)

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"✅  Saved to {out_path}")
    return out_path


def rename_columns(path: Path) -> None:
    """Rename UCI column names to our clean names and resave."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(path)

    # Lowercase all column names first
    df.columns = [c.lower() for c in df.columns]

    rename = {}
    for orig, new in COLUMN_MAP.items():
        if orig.lower() in df.columns:
            rename[orig.lower()] = new

    df.rename(columns=rename, inplace=True)

    # Keep only the columns we need (plus time info)
    keep = ["year", "month", "day", "hour"] + CFG.preprocessing.features
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    df.to_csv(path, index=False)
    print(f"✅  Columns renamed. Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")


if __name__ == "__main__":
    path = download()
    rename_columns(path)
    print("\n🎉  Data ready! Next step:")
    print("    python src/data/preprocessing.py")
