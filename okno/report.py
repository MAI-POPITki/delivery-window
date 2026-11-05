import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    """Markdown-таблица без зависимости от tabulate."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "/") for v in row) + " |")
    return "\n".join(out)
