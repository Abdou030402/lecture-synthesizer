# grouped_bars_by_document.py
# Creates grouped bar charts: x-axis = documents, bars = models.
# Outputs 4 figures per mode (GT & OCR): one for each metric.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- Paths relative to this script ---
BASE = Path(__file__).resolve().parent
GT_CSV = BASE / "per_doc_model_gt.csv"
OCR_CSV = BASE / "per_doc_model_ocr.csv"
OUTDIR = BASE / "charts_grouped_by_document"

METRICS = ["faithfulness", "helpfulness", "structure", "tts"]

def load_and_tidy(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERROR] CSV not found: {path}")
    df = pd.read_csv(path)
    # Remove 'mode' column, clean model names, ensure numeric metrics
    if "mode" in df.columns:
        df = df.drop(columns=["mode"])
    df["model"] = df["model"].astype(str).str.rstrip("_")
    for m in METRICS:
        df[m] = df[m].astype(float)
    # Order by document then model for stable plotting
    df = df[["document_id", "model"] + METRICS].sort_values(
        ["document_id", "model"]
    ).reset_index(drop=True)
    return df

import textwrap
import math
import matplotlib.pyplot as plt

def grouped_bars_by_document(df: pd.DataFrame, mode_name: str, outdir: Path) -> None:
    """
    For each metric, create a chart with:
      - x-axis: document_id
      - grouped bars: one bar per model per document
    Improves readability of x-tick labels by widening the figure, rotating,
    shrinking font, and wrapping long names.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    docs = sorted(df["document_id"].unique())
    models = sorted(df["model"].unique())

    # helper: wrap labels to multiple lines (e.g., every 10–12 chars)
    def wrap_label(s, width=10):
        return "\n".join(textwrap.wrap(str(s), width=width)) or str(s)

    # build wrapped labels up front
    wrapped_docs = [wrap_label(d, width=10) for d in docs]

    # figure width scales with number of documents (so labels have room)
    # tweak multiplier if needed
    fig_width = max(8.0, 0.75 * len(docs) + 2)

    for metric in METRICS:
        pivot = (
            df.pivot(index="document_id", columns="model", values=metric)
              .reindex(index=docs, columns=models)
        )

        plt.figure(figsize=(fig_width, 5.0))  # wider figure
        ax = pivot.plot(kind="bar")  # grouped bars

        ax.set_title(f"{mode_name}: {metric.capitalize()} by Document (Models as bars)")
        ax.set_xlabel("Document")
        ax.set_ylabel(metric.capitalize())

        # apply wrapped labels; rotate slightly & right-align
        ax.set_xticklabels(wrapped_docs, rotation=20, ha="right")
        ax.tick_params(axis="x", labelsize=8)  # smaller font for x labels
        ax.tick_params(axis="y", labelsize=9)

        # push legend above or outside to free x-axis space
        # (pick ONE of the two lines below)
        ax.legend(title="Model", ncol=min(len(models), 3), loc="upper center", bbox_to_anchor=(0.5, 1.15))
        # ax.legend(title="Model", loc="center left", bbox_to_anchor=(1.0, 0.5))  # alternative: outside right

        plt.tight_layout()
        outpath = outdir / f"{mode_name.lower()}_{metric}_grouped_by_document.png"
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()

def main():
    outdir = OUTDIR
    gt = load_and_tidy(GT_CSV)
    ocr = load_and_tidy(OCR_CSV)

    # One figure per metric for GT and OCR
    grouped_bars_by_document(gt, "GT", outdir)
    grouped_bars_by_document(ocr, "OCR", outdir)

    print(f"[OK] Saved grouped charts to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
