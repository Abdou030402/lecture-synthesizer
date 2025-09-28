# evaluation/extract_reasons.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import textwrap

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

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

def _normalize_0to1(x):
    try:
        v = float(x)
    except Exception:
        return np.nan
    if v < 0: 
        return np.nan
    if v > 1.5:  # assume 1–10
        v = v / 10.0
    return v

def load_eval(csv_path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["mode"] = mode

    # normalize numeric to 0–1
    hallu = df.get("hallucination_raw", df.get("hallucination_score", np.nan))
    help_ = df.get("helpfulness_raw", df.get("helpfulness_score", np.nan))
    struct = df.get("structure_raw", df.get("structure_score", np.nan))
    tts   = df.get("tts_raw", df.get("tts_score", np.nan))

    df["faithfulness"] = 1.0 - pd.Series(hallu).apply(_normalize_0to1)
    df["helpfulness"]  = pd.Series(help_).apply(_normalize_0to1)
    df["structure"]    = pd.Series(struct).apply(_normalize_0to1)
    df["tts"]          = pd.Series(tts).apply(_normalize_0to1)

    # map reasons
    df["faithfulness_reason"] = df.get("hallucination_reason", "")
    df["helpfulness_reason"]  = df.get("helpfulness_reason", "")
    df["structure_reason"]    = df.get("structure_reason", "")
    df["tts_reason"]          = df.get("tts_reason", "")
    return df

def trim(s: str, n=200):
    s = str(s).replace("\n", " ").strip()
    return (s[: n-3] + "...") if len(s) > n else s

def top_bottom(df, metric, k):
    sub = df[["document_id","model","prompt_style", metric, f"{metric}_reason"]].dropna()
    if sub.empty: 
        return pd.DataFrame(), pd.DataFrame()
    top = sub.sort_values(metric, ascending=False).head(k).copy()
    bot = sub.sort_values(metric, ascending=True).head(k).copy()
    top["rank_type"] = "top"
    bot["rank_type"] = "bottom"
    top["score"] = top[metric]; bot["score"] = bot[metric]
    top["reason"] = top[f"{metric}_reason"].apply(trim)
    bot["reason"] = bot[f"{metric}_reason"].apply(trim)
    cols = ["rank_type","model","document_id","prompt_style","score","reason"]
    return top[cols], bot[cols]

def main():
    ap = argparse.ArgumentParser(description="Extract concise judge reasons (top/bottom K per model/metric/mode).")
    ap.add_argument("--in-dir", default="evaluation", help="Folder with evaluation_results_gt.csv and evaluation_results_ocr.csv")
    ap.add_argument("--out-dir", default="evaluation/analysis_simple", help="Output root (tables/ & figures/ exist here)")
    ap.add_argument("--topk", type=int, default=3, help="How many top and bottom examples to keep per model×metric×mode")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    tabs = out_dir / "tables"
    ensure_dir(tabs)

    gt_csv, oc_csv = find_csvs(in_dir)
    dgt = load_eval(gt_csv, "gt")
    doc = load_eval(oc_csv, "ocr")

    metrics = ["faithfulness","helpfulness","structure","tts"]

    def build_samples(df, mode):
        rows = []
        for m in metrics:
            for mdl in sorted(df["model"].dropna().unique()):
                sub = df[df["model"]==mdl]
                top, bot = top_bottom(sub, m, args.topk)
                if not top.empty:
                    top.insert(0, "metric", m); top.insert(0, "mode", mode)
                    rows.append(top)
                if not bot.empty:
                    bot.insert(0, "metric", m); bot.insert(0, "mode", mode)
                    rows.append(bot)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    gt_samples = build_samples(dgt, "gt")
    oc_samples = build_samples(doc, "ocr")

    if not gt_samples.empty:
        gt_samples.to_csv(tabs / "reasons_samples_gt.csv", index=False)
    if not oc_samples.empty:
        oc_samples.to_csv(tabs / "reasons_samples_ocr.csv", index=False)

    # also write a compact Markdown to paste into thesis
    md_path = tabs / "reasons_samples.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Judge rationales: illustrative samples\n\n")
        for mode_label, samples in [("GT", gt_samples), ("OCR", oc_samples)]:
            if samples.empty:
                continue
            f.write(f"## {mode_label}\n\n")
            for m in metrics:
                s2 = samples[samples["metric"]==m]
                if s2.empty: 
                    continue
                f.write(f"### {m.capitalize()}\n\n")
                for mdl in sorted(s2["model"].unique()):
                    s3 = s2[s2["model"]==mdl]
                    if s3.empty: 
                        continue
                    f.write(f"**Model: {mdl}**\n\n")
                    # show top then bottom
                    for rank in ["top","bottom"]:
                        ss = s3[s3["rank_type"]==rank].sort_values("score", ascending=(rank=="bottom"))
                        if ss.empty: 
                            continue
                        f.write(f"- *{rank} examples*:\n")
                        for _, row in ss.iterrows():
                            f.write(f"  - doc `{row['document_id']}` (style `{row['prompt_style']}`), score={row['score']:.2f}: {row['reason']}\n")
                    f.write("\n")

    print("Saved:")
    if not gt_samples.empty: print(f"- {tabs/'reasons_samples_gt.csv'}")
    if not oc_samples.empty: print(f"- {tabs/'reasons_samples_ocr.csv'}")
    print(f"- {tabs/'reasons_samples.md'}")

if __name__ == "__main__":
    main()
