import os
import csv
import sys
import errno
from datetime import datetime

try:
    from deepeval.metrics import HallucinationMetric, GEval
except Exception:
    from deepeval.metrics.hallucination.hallucination import HallucinationMetric
    from deepeval.metrics.g_eval.g_eval import GEval

from deepeval.test_case import LLMTestCase, LLMTestCaseParams

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from nlp.system_prompts import SYSTEM_PROMPTS_MAP
except Exception:
    SYSTEM_PROMPTS_MAP = {}

RESULTS_ROOT = os.path.join("evaluation", "LLM_evaluation_results")
os.makedirs(RESULTS_ROOT, exist_ok=True)

GT_CSV = os.path.join(RESULTS_ROOT, "evaluation_results_gt.csv")
OCR_CSV = os.path.join(RESULTS_ROOT, "evaluation_results_ocr.csv")


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    lock_filename = os.path.join(RESULTS_ROOT, "evaluation.lock")
    if os.path.exists(lock_filename):
        try:
            with open(lock_filename, "r") as lf:
                pid_s = lf.read().strip()
        except Exception:
            pid_s = ""
        if pid_s:
            try:
                pid = int(pid_s)
            except ValueError:
                pid = None
            if pid:
                try:
                    os.kill(pid, 0)
                except OSError as e:
                    if e.errno == errno.ESRCH:
                        print(f"Stale lock (PID {pid}); removing.")
                        try:
                            os.remove(lock_filename)
                        except Exception as rem_err:
                            print(f"Remove lock manually: {rem_err}")
                            sys.exit(1)
                    else:
                        print(f"Another evaluation process is running (PID {pid}). Exiting.")
                        sys.exit(1)
            else:
                print("Invalid lock PID. Exiting to avoid conflicts.")
                sys.exit(1)
    try:
        with open(lock_filename, "x") as lf:
            lf.write(str(os.getpid()))
    except FileExistsError:
        print("Could not acquire lock; another evaluation is running.")
        sys.exit(1)

    try:
        header = [
            "document_id", "model", "prompt_style",
            "hallucination_score", "hallucination_reason",
            "helpfulness_score", "helpfulness_reason",
            "structure_score", "structure_reason",
            "tts_score", "tts_reason"
        ]
        for csv_path in (GT_CSV, OCR_CSV):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

        MODES = [
            ("gt", GT_CSV, os.path.join("ocr_ground_truths")),
            ("ocr", OCR_CSV, os.path.join("OCR_outputs", "trocr_craft")),
        ]

        for mode, csv_path, notes_dir in MODES:
            base_dir = os.path.join("nlp_outputs", mode)
            if not os.path.isdir(base_dir):
                print(f"Directory '{base_dir}' not found, skipping {mode.upper()}.")
                continue

            with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)

                for doc_id in sorted(os.listdir(base_dir)):
                    doc_path = os.path.join(base_dir, doc_id)
                    if not os.path.isdir(doc_path):
                        continue

                    notes_path = os.path.join(notes_dir, f"{doc_id}.txt")
                    try:
                        with open(notes_path, "r", encoding="utf-8") as nf:
                            notes_text = nf.read().strip()
                    except FileNotFoundError:
                        print(f"[{mode}] Missing notes for {doc_id}: {notes_path}")
                        continue

                    log_dir = os.path.join(RESULTS_ROOT, mode, doc_id)
                    os.makedirs(log_dir, exist_ok=True)

                    for filename in sorted(os.listdir(doc_path)):
                        if not filename.lower().endswith(".txt"):
                            continue

                        file_path = os.path.join(doc_path, filename)
                        try:
                            with open(file_path, "r", encoding="utf-8") as lf:
                                lecture_text = lf.read().strip()
                        except Exception as e:
                            print(f"[{mode}] Cannot read {file_path}: {e}")
                            continue

                        name_part = filename[:-4]
                        model_name, prompt_style = None, None
                        for style_key in SYSTEM_PROMPTS_MAP.keys():
                            suffix = f"_{style_key}"
                            if name_part.endswith(suffix):
                                prompt_style = style_key
                                model_name = name_part[: -len(suffix)]
                                break
                        if model_name is None or prompt_style is None:
                            parts = name_part.split("_", 1)
                            if len(parts) == 2:
                                model_name, prompt_style = parts[0].strip(), parts[1].strip()
                            else:
                                model_name, prompt_style = name_part.strip(), ""

                        system_prompt_text = SYSTEM_PROMPTS_MAP.get(prompt_style, "")

                        tc_notes = LLMTestCase(
                            input=notes_text,
                            actual_output=lecture_text,
                            context=[notes_text]
                        )

                        tc_output_only = LLMTestCase(
                            input=notes_text, 
                            actual_output=lecture_text
                        )

                        tc_tts = LLMTestCase(
                            input=notes_text,
                            actual_output=lecture_text,
                            context=[system_prompt_text] if system_prompt_text else []
                        )

                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        base = os.path.join(log_dir, f"{doc_id}_{model_name}_{prompt_style}_{stamp}")
                        try:
                            write_text(base + "_notes.txt", notes_text)
                            write_text(base + "_system_prompt.txt", system_prompt_text or "(empty)")
                            write_text(base + "_output.txt", lecture_text)
                        except Exception:
                            pass

                        halluc_score = help_score = struct_score = tts_score = -1.0
                        halluc_reason = help_reason = struct_reason = tts_reason = "Evaluation failed"

                        try:
                            halu = HallucinationMetric()
                            halu.measure(tc_notes)
                            if halu.score is not None:
                                halluc_score = float(halu.score)
                            halluc_reason = (halu.reason or "").replace("\n", " ")
                        except Exception as e:
                            print(f"[{mode}] Hallucination failed ({doc_id} | {filename}): {e}")

                        try:
                            help_metric = GEval(
                                name="Helpfulness",
                                criteria=(
                                    "Judge how helpful, thorough, and educational ACTUAL_OUTPUT is for teaching the content implied by CONTEXT (the source notes). "
                                    "Consider coverage of key points, clarity of explanation, and usefulness to students. "
                                    "Return a numeric score in [0,1] and a brief reason."
                                ),
                                evaluation_params=[LLMTestCaseParams.CONTEXT, LLMTestCaseParams.ACTUAL_OUTPUT],
                            )
                            help_metric.measure(tc_notes)
                            if help_metric.score is not None:
                                help_score = float(help_metric.score)
                            help_reason = (help_metric.reason or "").replace("\n", " ")
                        except Exception as e:
                            print(f"[{mode}] Helpfulness failed ({doc_id} | {filename}): {e}")

                        try:
                            struct_metric = GEval(
                                name="StructureClarity",
                                criteria=(
                                    "Evaluate organization and clarity of ACTUAL_OUTPUT as a spoken lecture. "
                                    "Consider logical flow, signposting/sections, transitions, and ease of following explanations. "
                                    "Return a numeric score in [0,1] and a brief reason."
                                ),
                                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                            )
                            struct_metric.measure(tc_output_only)
                            if struct_metric.score is not None:
                                struct_score = float(struct_metric.score)
                            struct_reason = (struct_metric.reason or "").replace("\n", " ")
                        except Exception as e:
                            print(f"[{mode}] Structure failed ({doc_id} | {filename}): {e}")

                        try:
                            if system_prompt_text:
                                tts_metric = GEval(
                                    name="TTSCompliance",
                                    criteria=(
                                        "Evaluate whether ACTUAL_OUTPUT complies with the system prompt's TTS style/formatting guidance in CONTEXT. "
                                        "Consider tone, avoidance of disallowed elements, and any formatting/tag rules. "
                                        "Return a numeric score in [0,1] and a brief reason."
                                    ),
                                    evaluation_params=[LLMTestCaseParams.CONTEXT, LLMTestCaseParams.ACTUAL_OUTPUT],
                                )
                                tts_metric.measure(tc_tts)
                                if tts_metric.score is not None:
                                    tts_score = float(tts_metric.score)
                                tts_reason = (tts_metric.reason or "").replace("\n", " ")
                            else:
                                tts_score, tts_reason = -1.0, "No system prompt available."
                        except Exception as e:
                            print(f"[{mode}] TTS compliance failed ({doc_id} | {filename}): {e}")

                        writer.writerow([
                            doc_id,
                            model_name,
                            prompt_style,
                            f"{halluc_score:.3f}", halluc_reason,
                            f"{help_score:.3f}", help_reason,
                            f"{struct_score:.3f}", struct_reason,
                            f"{tts_score:.3f}", tts_reason
                        ])
                        fcsv.flush()

                        flags = [halluc_score, help_score, struct_score, tts_score]
                        if all(s >= 0.0 for s in flags if s is not None):
                            print(f"{mode.upper()} | {doc_id} | {filename} -> OK")
                        else:
                            print(f"{mode.UPPER()} | {doc_id} | {filename} -> some errors")

    finally:
        try:
            os.remove(lock_filename)
        except Exception as e:
            print(f"Warning: could not remove lock file: {e}")


if __name__ == "__main__":
    main()
