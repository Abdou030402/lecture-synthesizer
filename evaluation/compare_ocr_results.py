import os
import pandas as pd
import matplotlib.pyplot as plt

def load_results(engine_folder):
    path = os.path.join("evaluation", "evaluation_results", engine_folder, "results.csv")
    df = pd.read_csv(path)
    df["Engine"] = engine_folder
    return df

def compare_models(engine_folders, output_dir="evaluation/final_comparison_results", excluded_engines=None):
    os.makedirs(output_dir, exist_ok=True)
    combined_rows = []
    avg_summary = []
    typewise_summary = []

    for engine in engine_folders:
        df = load_results(engine)
        per_file = df[~df["File"].isin(["Average", "Handwritten Average", "Printed Average"])]
        combined_rows.append(per_file)

        avg_row = df[df["File"] == "Average"]
        if not avg_row.empty:
            avg_summary.append({
                "Engine": engine,
                "WER": float(avg_row["WER"].values[0]) * 100,
                "CER": float(avg_row["CER"].values[0]) * 100
            })

        hand_row = df[df["File"] == "Handwritten Average"]
        if not hand_row.empty:
            typewise_summary.append({
                "Engine": engine,
                "Type": "Handwritten",
                "WER": float(hand_row["WER"].values[0]) * 100,
                "CER": float(hand_row["CER"].values[0]) * 100
            })

        print_row = df[df["File"] == "Printed Average"]
        if not print_row.empty:
            typewise_summary.append({
                "Engine": engine,
                "Type": "Printed",
                "WER": float(print_row["WER"].values[0]) * 100,
                "CER": float(print_row["CER"].values[0]) * 100
            })

    combined_df = pd.concat(combined_rows)
    summary_df = pd.DataFrame(avg_summary)
    typewise_df = pd.DataFrame(typewise_summary)

    summary_df.to_csv(os.path.join(output_dir, "comparison_summary.csv"), index=False)
    typewise_df.to_csv(os.path.join(output_dir, "typewise_comparison_summary.csv"), index=False)

    if excluded_engines:
        chart_summary_df = summary_df[~summary_df["Engine"].isin(excluded_engines)]
        chart_typewise_df = typewise_df[~typewise_df["Engine"].isin(excluded_engines)]
        chart_combined_df = combined_df[~combined_df["Engine"].isin(excluded_engines)]
    else:
        chart_summary_df = summary_df
        chart_typewise_df = typewise_df
        chart_combined_df = combined_df

    plt.figure(figsize=(6, 5))
    plt.bar(chart_summary_df["Engine"], chart_summary_df["WER"], color="skyblue")
    plt.ylabel("Average WER (%)")
    plt.title("Overall WER Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_wer_comparison.png"))
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(chart_summary_df["Engine"], chart_summary_df["CER"], color="lightcoral")
    plt.ylabel("Average CER (%)")
    plt.title("Overall CER Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_cer_comparison.png"))
    plt.close()

    pivot_wer = chart_combined_df.pivot(index="File", columns="Engine", values="WER") * 100
    pivot_cer = chart_combined_df.pivot(index="File", columns="Engine", values="CER") * 100

    pivot_wer.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("WER (%)")
    plt.title("WER per File - All Engines")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wer_per_file_all_engines.png"))
    plt.close()

    pivot_cer.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("CER (%)")
    plt.title("CER per File - All Engines")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cer_per_file_all_engines.png"))
    plt.close()

    for metric in ["WER", "CER"]:
        plt.figure(figsize=(7, 5))
        for t in ["Handwritten", "Printed"]:
            subset = chart_typewise_df[chart_typewise_df["Type"] == t]
            plt.bar(subset["Engine"], subset[metric], label=t)
        plt.ylabel(f"{metric} (%)")
        plt.title(f"{metric} by Document Type")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric.lower()}_type_comparison.png"))
        plt.close()

    print(f"✅ Comparison complete.\nResults saved to: {output_dir}")

if __name__ == "__main__":
    engine_folders = ["trocr_craft", "PaddleOCR", "Tesseract"]
    excluded_engines = ["Tesseract"]
    compare_models(engine_folders, excluded_engines=excluded_engines)
