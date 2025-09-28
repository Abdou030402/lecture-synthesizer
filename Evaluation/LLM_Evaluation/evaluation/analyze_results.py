# evaluation/analyze_results_simple.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Utilities
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_float(x):
    try:
        if x is None:
            return np.nan
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def find_csvs(in_dir: Path):
    gt = in_dir / "evaluation_results_gt.csv"
    oc = in_dir / "evaluation_results_ocr.csv"
    if gt.exists() and oc.exists():
        return gt, oc
    sub = in_dir / "LLM_evaluation_results"
    gt2 = sub / "evaluation_results_gt.csv"
    oc2 = sub / "evaluation_results_ocr.csv"
    if gt2.exists() and oc2.exists():
        return gt2, oc2
    raise FileNotFoundError(f"Couldn't find CSVs in {in_dir} or {sub}")

def _normalize_0to1(series: pd.Series) -> pd.Series:
    """If any value > 1.5, assume it's 1–10 and divide by 10."""
    ser = series.astype(float)
    if ser.dropna().gt(1.5).any():
        ser = ser / 10.0
    return ser

def load_and_prepare(csv_path: Path, mode_tag: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["mode"] = mode_tag

    # hallucination -> faithfulness
    if "hallucination_raw" in df.columns:
        hallu_raw = _normalize_0to1(df["hallucination_raw"].apply(safe_float))
    else:
        hallu_raw = _normalize_0to1(df.get("hallucination_score", pd.Series(np.nan)).apply(safe_float))

    def get_raw(name):
        if f"{name}_raw" in df.columns:
            return _normalize_0to1(df[f"{name}_raw"].apply(safe_float))
        return _normalize_0to1(df.get(f"{name}_score", pd.Series(np.nan)).apply(safe_float))

    help_raw = get_raw("helpfulness")
    struct_raw = get_raw("structure")
    tts_raw   = get_raw("tts")

    out = pd.DataFrame({
        "document_id": df.get("document_id"),
        "model": df.get("model"),
        "prompt_style": df.get("prompt_style"),
        "mode": mode_tag,
        "faithfulness": 1.0 - hallu_raw,
        "helpfulness": help_raw,
        "structure": struct_raw,
        "tts": tts_raw,
    })

    for c in ["faithfulness","helpfulness","structure","tts"]:
        out.loc[(out[c] < 0) | (out[c] > 1), c] = np.nan
    return out

# -------------------------
# Simple plots
# -------------------------

def bar_gt_vs_ocr(per_model_gt: pd.DataFrame, per_model_ocr: pd.DataFrame, metric: str, out_png: Path, title_suffix=""):
    """Bar chart: x=models, bars=GT vs OCR mean for given metric."""
    if per_model_gt.empty and per_model_ocr.empty:
        return
    models = sorted(set(per_model_gt["model"].dropna().unique()).union(set(per_model_ocr["model"].dropna().unique())))
    if not models:
        return
    gt_means = [per_model_gt.loc[per_model_gt["model"]==m, metric].mean() for m in models]
    oc_means = [per_model_ocr.loc[per_model_ocr["model"]==m, metric].mean() for m in models]

    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(8,4))
    plt.bar(x - width/2, gt_means, width, label="GT")
    plt.bar(x + width/2, oc_means, width, label="OCR")
    plt.xticks(x, models, rotation=0)
    plt.ylabel(f"{metric} (0–1)")
    title = f"{metric.capitalize()} — GT vs OCR by model"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def bar_deltas(delta_df: pd.DataFrame, metric: str, out_png: Path):
    """Bar chart of OCR−GT delta per model for a metric (positive = OCR better)."""
    if delta_df.empty:
        return
    models = list(delta_df["model"])
    vals = list(delta_df[f"delta_{metric}"])
    x = np.arange(len(models))
    plt.figure(figsize=(7,4))
    plt.bar(x, vals)
    plt.xticks(x, models, rotation=0)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.ylabel(f"Δ {metric} (OCR − GT)")
    plt.title(f"Robustness to OCR noise — Δ {metric} by model")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def wins_bar(wins_df: pd.DataFrame, metric: str, out_png: Path, mode_label: str):
    """Bar chart of win counts across documents for a metric."""
    if wins_df.empty: 
        return
    row = wins_df[wins_df["metric"]==metric]
    if row.empty:
        return
    row = row.iloc[0].drop(labels=["metric"])
    models = list(row.index)
    counts = list(row.values.astype(float))
    x = np.arange(len(models))
    plt.figure(figsize=(6,4))
    plt.bar(x, counts)
    plt.xticks(x, models, rotation=0)
    plt.ylabel("Doc wins (count)")
    plt.title(f"{mode_label} — Document wins for {metric}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Simple DeepEval analysis: per-doc & per-model means + wins + robustness.")
    ap.add_argument("--in-dir", default="evaluation", help="Folder containing evaluation_results_gt.csv and evaluation_results_ocr.csv")
    ap.add_argument("--out-dir", default="evaluation/analysis_simple", help="Output folder for tables and figures")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    figs = out_dir / "figs"
    tabs = out_dir / "tables"
    ensure_dir(figs); ensure_dir(tabs)

    gt_csv, oc_csv = find_csvs(in_dir)
    df_gt  = load_and_prepare(gt_csv, "gt")
    df_ocr = load_and_prepare(oc_csv, "ocr")

    # -------- per document × model (avg over prompt_style) --------
    group_keys = ["mode","document_id","model"]
    metrics = ["faithfulness","helpfulness","structure","tts"]

    per_doc_model_gt  = (df_gt.groupby(group_keys, dropna=False)[metrics].mean().reset_index())
    per_doc_model_ocr = (df_ocr.groupby(group_keys, dropna=False)[metrics].mean().reset_index())

    per_doc_model_gt[metrics]  = per_doc_model_gt[metrics].round(2)
    per_doc_model_ocr[metrics] = per_doc_model_ocr[metrics].round(2)

    per_doc_model_gt.to_csv(tabs / "per_doc_model_gt.csv", index=False)
    per_doc_model_ocr.to_csv(tabs / "per_doc_model_ocr.csv", index=False)

    # -------- per model overall means (within each mode) ----------
    per_model_gt  = (df_gt.groupby(["mode","model"], dropna=False)[metrics].mean().reset_index())
    per_model_ocr = (df_ocr.groupby(["mode","model"], dropna=False)[metrics].mean().reset_index())

    # composite
    per_model_gt["composite"]  = per_model_gt[metrics].mean(axis=1)
    per_model_ocr["composite"] = per_model_ocr[metrics].mean(axis=1)

    per_model_gt[metrics + ["composite"]]  = per_model_gt[metrics + ["composite"]].round(2)
    per_model_ocr[metrics + ["composite"]] = per_model_ocr[metrics + ["composite"]].round(2)

    per_model_gt.to_csv(tabs / "model_means_gt.csv", index=False)
    per_model_ocr.to_csv(tabs / "model_means_ocr.csv", index=False)

    # -------- overall means by mode (single row per mode) ---------
    overall_gt  = df_gt[metrics].mean().to_frame().T
    overall_gt.insert(0, "mode", "gt")
    overall_ocr = df_ocr[metrics].mean().to_frame().T
    overall_ocr.insert(0, "mode", "ocr")
    overall = pd.concat([overall_gt, overall_ocr], ignore_index=True)
    overall["composite"] = overall[metrics].mean(axis=1)
    overall = overall.round(2)
    overall.to_csv(tabs / "overall_by_mode.csv", index=False)

    # -------- simple GT vs OCR bar charts per metric --------------
    for metric in metrics:
        bar_gt_vs_ocr(per_model_gt, per_model_ocr, metric, figs / f"bars_gt_vs_ocr_{metric}.png")

    # -------- NEW: composite GT vs OCR bar chart ------------------
    bar_gt_vs_ocr(per_model_gt, per_model_ocr, "composite", figs / "bars_gt_vs_ocr_composite.png", title_suffix="overall")

    # -------- NEW: robustness (OCR − GT deltas) -------------------
    # merge per-model tables and compute deltas
    merged = pd.merge(
        per_model_gt[["model"] + metrics + ["composite"]],
        per_model_ocr[["model"] + metrics + ["composite"]],
        on="model",
        suffixes=("_gt","_ocr")
    )
    for m in metrics + ["composite"]:
        merged[f"delta_{m}"] = merged[f"{m}_ocr"] - merged[f"{m}_gt"]
    merged.to_csv(tabs / "model_deltas_gt_ocr.csv", index=False)

    # delta bar charts (positive = OCR better)
    for m in metrics + ["composite"]:
        bar_deltas(merged[["model", f"delta_{m}"]], m, figs / f"bars_delta_{m}.png")

    # -------- NEW: wins across documents --------------------------
    def wins_table(per_doc):
        rows = []
        for m in metrics:
            winners = per_doc.loc[per_doc.groupby("document_id")[m].idxmax()][["document_id","model",m]]
            counts = winners["model"].value_counts().to_dict()
            rows.append({"metric": m, **counts})
        return pd.DataFrame(rows).fillna(0)

    wins_gt = wins_table(per_doc_model_gt)
    wins_ocr = wins_table(per_doc_model_ocr)
    wins_gt.to_csv(tabs / "wins_by_metric_gt.csv", index=False)
    wins_ocr.to_csv(tabs / "wins_by_metric_ocr.csv", index=False)

    for m in metrics:
        wins_bar(wins_gt, m, figs / f"wins_gt_{m}.png", mode_label="GT")
        wins_bar(wins_ocr, m, figs / f"wins_ocr_{m}.png", mode_label="OCR")

    print("\nDone. Outputs:")
    print(f"- Tables: {tabs}")
    print("  • per_doc_model_gt.csv / per_doc_model_ocr.csv (avg over prompts)")
    print("  • model_means_gt.csv / model_means_ocr.csv (incl. composite)")
    print("  • overall_by_mode.csv (incl. composite)")
    print("  • model_deltas_gt_ocr.csv (OCR−GT per metric + composite)")
    print("  • wins_by_metric_gt.csv / wins_by_metric_ocr.csv")
    print(f"- Figures: {figs}")
    print("  • bars_gt_vs_ocr_*.png (faithfulness/helpfulness/structure/tts/composite)")
    print("  • bars_delta_*.png (robustness deltas per metric + composite)")
    print("  • wins_gt_*.png and wins_ocr_*.png (document wins per metric)")
    print("")
    print("Interpretation: higher is better; positive delta means OCR > GT.")
    
if __name__ == "__main__":
    main()
