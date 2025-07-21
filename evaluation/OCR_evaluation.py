import os
import csv
import matplotlib.pyplot as plt
from jiwer import wer, cer

def compare_folders(gt_dir: str, ocr_dir: str):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(".txt")])
    ocr_files = sorted([f for f in os.listdir(ocr_dir) if f.lower().endswith(".txt")])
    common_files = set(gt_files) & set(ocr_files)

    if not common_files:
        print("No matching files found between the two directories.")
        return

    total_wer = total_cer = 0.0
    count = 0

    handwritten_wer = handwritten_cer = 0.0
    handwritten_count = 0

    printed_wer = printed_cer = 0.0
    printed_count = 0

    results = []

    print(f"{'File':40s} {'WER':>10s} {'CER':>10s}")
    print("-" * 65)

    for filename in sorted(common_files):
        gt_path = os.path.join(gt_dir, filename)
        ocr_path = os.path.join(ocr_dir, filename)

        with open(gt_path, 'r', encoding='utf-8') as f:
            reference = f.read().strip()
        with open(ocr_path, 'r', encoding='utf-8') as f:
            hypothesis = f.read().strip()

        file_wer = wer(reference, hypothesis)
        file_cer = cer(reference, hypothesis)

        total_wer += file_wer
        total_cer += file_cer
        count += 1

        if filename.startswith("handwritten"):
            handwritten_wer += file_wer
            handwritten_cer += file_cer
            handwritten_count += 1
        elif filename.startswith("printed"):
            printed_wer += file_wer
            printed_cer += file_cer
            printed_count += 1

        results.append((filename, file_wer, file_cer))
        print(f"{filename:40s} {file_wer:.2%} {file_cer:.2%}")

    avg_wer = total_wer / count if count else 0
    avg_cer = total_cer / count if count else 0

    avg_hand_wer = handwritten_wer / handwritten_count if handwritten_count else 0
    avg_hand_cer = handwritten_cer / handwritten_count if handwritten_count else 0

    avg_print_wer = printed_wer / printed_count if printed_count else 0
    avg_print_cer = printed_cer / printed_count if printed_count else 0

    print("\nOverall:")
    print(f"Average WER: {avg_wer:.2%}")
    print(f"Average CER: {avg_cer:.2%}")
    print(f"\nHandwritten Average WER: {avg_hand_wer:.2%}")
    print(f"Handwritten Average CER: {avg_hand_cer:.2%}")
    print(f"\nPrinted Average WER: {avg_print_wer:.2%}")
    print(f"Printed Average CER: {avg_print_cer:.2%}")

    engine_name = os.path.basename(ocr_dir)
    output_dir = os.path.join("evaluation", "evaluation_results", engine_name)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File", "WER", "CER"])
        for filename, file_wer, file_cer in results:
            writer.writerow([filename, f"{file_wer:.4f}", f"{file_cer:.4f}"])
        writer.writerow(["Average", f"{avg_wer:.4f}", f"{avg_cer:.4f}"])
        writer.writerow(["Handwritten Average", f"{avg_hand_wer:.4f}", f"{avg_hand_cer:.4f}"])
        writer.writerow(["Printed Average", f"{avg_print_wer:.4f}", f"{avg_print_cer:.4f}"])

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Engine: {engine_name}\n")
        f.write(f"Overall WER: {avg_wer:.2%}\n")
        f.write(f"Overall CER: {avg_cer:.2%}\n\n")
        f.write(f"Handwritten WER: {avg_hand_wer:.2%}\n")
        f.write(f"Handwritten CER: {avg_hand_cer:.2%}\n\n")
        f.write(f"Printed WER: {avg_print_wer:.2%}\n")
        f.write(f"Printed CER: {avg_print_cer:.2%}\n")

    files = [r[0] for r in results]
    wers = [r[1] * 100 for r in results]
    cers = [r[2] * 100 for r in results]

    plt.figure(figsize=(10, 5))
    plt.barh(files, wers)
    plt.xlabel("WER (%)")
    plt.title(f"Word Error Rate per File - {engine_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wer_chart.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.barh(files, cers)
    plt.xlabel("CER (%)")
    plt.title(f"Character Error Rate per File - {engine_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cer_chart.png"))
    plt.close()

    categories = ['Handwritten', 'Printed']
    avg_wers = [avg_hand_wer * 100, avg_print_wer * 100]
    avg_cers = [avg_hand_cer * 100, avg_print_cer * 100]

    plt.figure(figsize=(6, 5))
    plt.bar(categories, avg_wers, color=['#1f77b4', '#ff7f0e'])
    plt.ylabel("Average WER (%)")
    plt.title(f"WER Comparison - {engine_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "handwritten_vs_printed_wer.png"))
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(categories, avg_cers, color=['#1f77b4', '#ff7f0e'])
    plt.ylabel("Average CER (%)")
    plt.title(f"CER Comparison - {engine_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "handwritten_vs_printed_cer.png"))
    plt.close()

    print(f"\nEvaluation results saved to: {output_dir}")

if __name__ == "__main__":
    gt_folder = "ocr_ground_truths"
    ocr_output_folder = "OCR_outputs/PaddleOCR"
    compare_folders(gt_folder, ocr_output_folder)
