from pathlib import Path
import pandas as pd

def load_spam_csv(path: str = "data/spam.csv") -> pd.DataFrame:
    """Load the spam CSV, return DataFrame with columns ['label','text']."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p.resolve()}")
    df = pd.read_csv(p, encoding="latin-1", usecols=["v1", "v2"])
    df.columns = ["label", "text"]
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].str.strip()
    return df
