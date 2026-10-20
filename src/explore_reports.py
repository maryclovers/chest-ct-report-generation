# explore_reports.py
# looks at the grounded report sentences so I understand the target text the
# model has to generate. produces a summary file and a bar chart of which
# anatomical regions appear most, both saved to outputs.

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, OUTPUTS_DIR

import pandas as pd
import matplotlib.pyplot as plt


def load_reports():
    # find the parquet that download_data.py saved and load it
    parquet_files = sorted(DATA_DIR.glob("grounded_reports_*.parquet"))
    if not parquet_files:
        print("no grounded reports parquet found in data/.")
        print("run src/download_data.py first.")
        sys.exit(1)
    path = parquet_files[0]
    print(f"loading {path.name} ...")
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    return df


def summarise(df):
    # basic shape and content checks, written to screen and a summary file
    lines = []
    lines.append("=" * 60)
    lines.append("GROUNDED REPORTS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"total rows:        {len(df)}")
    lines.append(f"columns:           {list(df.columns)}")
    lines.append(f"unique volumes:    {df['Volumename'].nunique()}")
    lines.append(f"unique anatomy:    {df['Anatomy'].nunique()}")

    # how many rows are grounded to a lung region (anatomy label starts with 'lung')
    lung_mask = df["Anatomy"].fillna("").str.startswith("lung")
    lines.append(f"lung-grounded rows: {lung_mask.sum()}")

    # average sentence length in words, a quick sense of how long targets are
    word_counts = df["Sentence"].fillna("").str.split().apply(len)
    lines.append(f"mean words/sentence: {word_counts.mean():.1f}")

    lines.append("")
    lines.append("top 15 anatomy regions by row count:")
    top = df["Anatomy"].value_counts().head(15)
    for region, count in top.items():
        lines.append(f"  {str(region):<35} {count}")

    lines.append("")
    lines.append("example lung-grounded sentences:")
    examples = df[lung_mask]["Sentence"].dropna().head(3)
    for i, sentence in enumerate(examples, 1):
        # trim very long reports so the summary stays readable
        short = sentence if len(sentence) < 200 else sentence[:200] + " ..."
        lines.append(f"  {i}. {short}")

    text = "\n".join(lines)
    print(text)

    out_path = OUTPUTS_DIR / "grounded_reports_summary.txt"
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print("")
    print(f"saved summary to: {out_path}")


def plot_top_regions(df, n=15):
    # horizontal bar chart of the most common anatomy regions
    top = df["Anatomy"].value_counts().head(n).iloc[::-1]  # reverse so biggest is on top

    plt.figure(figsize=(10, 7))
    plt.barh(top.index.astype(str), top.values, color="steelblue")
    plt.xlabel("number of grounded sentences")
    plt.title(f"top {n} anatomical regions in RadGenome grounded reports")
    plt.tight_layout()

    out_path = OUTPUTS_DIR / "anatomy_region_counts.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved chart to: {out_path}")


def main():
    df = load_reports()
    summarise(df)
    plot_top_regions(df)


if __name__ == "__main__":
    main()
