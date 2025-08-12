#currently implemented to work only for the first example
#still need to add a working openai api key to .env

import os
import json
import csv
import time
import openai
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nlp.system_prompts import SYSTEM_PROMPTS_MAP

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

EVAL_PROMPTS = {
    "hallucination": (
        "You are an expert evaluator of lecture content. "
        "Evaluate the model's lecture output for *hallucinations or factual inaccuracies*. "
        "The model was given the following system instructions:\n\"\"\"{system_prompt}\"\"\"\n\n"
        "Lecture Notes (Ground Truth Input):\n\"\"\"{notes}\"\"\"\n\n"
        "Model's Lecture Output:\n\"\"\"{output}\"\"\"\n\n"
        "**Hallucination Evaluation**: Determine if the output contains any information that is not supported by the notes or any made-up facts. "
        "Rate the output on a scale of 1 to 10 (10 = no hallucination at all, completely factual; 1 = severely hallucinates or contradicts the notes). "
        "Provide a concise reason for the score.\n"
        "Respond ONLY in JSON format as {{\"score\": <number>, \"reason\": \"<explanation>\"}}."
    ),
    "helpfulness": (
        "You are an expert evaluator of lecture content. "
        "Evaluate the model's lecture output for *helpfulness and educational value*. "
        "The model was given the following system instructions:\n\"\"\"{system_prompt}\"\"\"\n\n"
        "Lecture Notes (Input):\n\"\"\"{notes}\"\"\"\n\n"
        "Model's Lecture Output:\n\"\"\"{output}\"\"\"\n\n"
        "**Helpfulness Evaluation**: Determine how useful, thorough, and informative the lecture output is in teaching the content of the notes. "
        "Consider if it covers the important points, explains them well, and would be valuable to a student. "
        "Rate the helpfulness on a scale of 1 to 10 (10 = extremely helpful and informative; 1 = not helpful or missing the point). "
        "Provide a concise reason for the score.\n"
        "Respond ONLY in JSON format as {{\"score\": <number>, \"reason\": \"<explanation>\"}}."
    ),
    "structure": (
        "You are an expert evaluator of lecture content. "
        "Evaluate the model's lecture output for *structure and clarity*. "
        "The model was given the following system instructions:\n\"\"\"{system_prompt}\"\"\"\n\n"
        "Lecture Notes (Input):\n\"\"\"{notes}\"\"\"\n\n"
        "Model's Lecture Output:\n\"\"\"{output}\"\"\"\n\n"
        "**Structure & Clarity Evaluation**: Analyze how well the lecture is organized and how clearly it is written. "
        "Consider the logical flow of topics, the presence of sections or a clear narrative, and the clarity of explanations. "
        "Rate the structure & clarity on a scale of 1 to 10 (10 = excellently structured and very clear; 1 = very disorganized or confusing). "
        "Provide a concise reason for the score.\n"
        "Respond ONLY in JSON format as {{\"score\": <number>, \"reason\": \"<explanation>\"}}."
    ),
    "tts": (
        "You are an expert evaluator of lecture content. "
        "Evaluate the model's lecture output for *compliance with the text-to-speech (TTS) prompt instructions*. "
        "Below is the system prompt given to the model:\n\"\"\"{system_prompt}\"\"\"\n\n"
        "Lecture Notes (Input):\n\"\"\"{notes}\"\"\"\n\n"
        "Model's Lecture Output:\n\"\"\"{output}\"\"\"\n\n"
        "**TTS Prompt Compliance Evaluation**: Determine whether the output followed **all the instructions and guidelines** from the system prompt, especially those related to style or formatting for text-to-speech. "
        "This includes adhering to tone, avoiding disallowed content, and any formatting requirements. "
        "Rate the compliance on a scale of 1 to 10 (10 = fully compliant with the prompt; 1 = significantly deviates from instructions). "
        "Provide a concise reason for the score.\n"
        "Respond ONLY in JSON format as {{\"score\": <number>, \"reason\": \"<explanation>\"}}."
    )
}

def evaluate_metric(metric_key, system_prompt, notes_text, output_text, log_dir, doc_id, model_name, prompt_style, max_retries=3):
    prompt_content = EVAL_PROMPTS[metric_key].format(
        system_prompt=system_prompt,
        notes=notes_text,
        output=output_text
    )
    messages = [{"role": "user", "content": prompt_content}]

    os.makedirs(log_dir, exist_ok=True)
    base_filename = f"{doc_id}_{model_name}_{prompt_style}_{metric_key}"
    prompt_path = os.path.join(log_dir, f"{base_filename}_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as pf:
        pf.write(prompt_content)

    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )
        except openai.error.RateLimitError:
            time.sleep(5)
            continue
        except openai.error.OpenAIError as e:
            return None, f"OpenAI Error: {e}"

        content = response.choices[0].message.content.strip()

        response_path = os.path.join(log_dir, f"{base_filename}_response.txt")
        with open(response_path, "w", encoding="utf-8") as rf:
            rf.write(content)

        if content.startswith("```"):
            content = content.strip("```").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            continue

        if isinstance(result, dict) and "score" in result and "reason" in result:
            score = int(float(result["score"])) if isinstance(result["score"], (int, float, str)) else -1
            reason = str(result["reason"])
            return score, reason

    return None, "Evaluation failed or malformed response"


gt_csv_path = "evaluation_results_gt.csv"
ocr_csv_path = "evaluation_results_ocr.csv"

with open(gt_csv_path, mode="w", newline="", encoding="utf-8") as f_gt, \
     open(ocr_csv_path, mode="w", newline="", encoding="utf-8") as f_ocr:
    writer_gt = csv.writer(f_gt)
    writer_ocr = csv.writer(f_ocr)
    header = [
        "document_id", "model", "prompt_style",
        "hallucination_score", "hallucination_reason",
        "helpfulness_score", "helpfulness_reason",
        "structure_score", "structure_reason",
        "tts_score", "tts_reason"
    ]
    writer_gt.writerow(header)
    writer_ocr.writerow(header)

    for mode, writer in [("gt", writer_gt), ("ocr", writer_ocr)]:
        base_dir = os.path.join("nlp_outputs", mode)
        if mode == "gt":
            notes_dir = os.path.join("ocr_ground_truths")
        else:
            notes_dir = os.path.join("OCR_outputs", "trocr_craft")

        if not os.path.isdir(base_dir):
            print(f"Directory '{base_dir}' not found, skipping...")
            continue

        for doc_id in sorted(os.listdir(base_dir)):
            doc_path = os.path.join(base_dir, doc_id)
            if not os.path.isdir(doc_path):
                continue 
            notes_path = os.path.join(notes_dir, f"{doc_id}.txt")
            try:
                with open(notes_path, "r", encoding="utf-8") as nf:
                    notes_text = nf.read().strip()
            except FileNotFoundError:
                print(f"Warning: Notes file not found for {mode.upper()} document {doc_id} ({notes_path}). Skipping this document.")
                continue

            for filename in sorted(os.listdir(doc_path)):
                if not filename.lower().endswith(".txt"):
                    continue
                file_path = os.path.join(doc_path, filename)

                name_part = filename[:-4]
                model_name = None
                prompt_style = None

                for style in SYSTEM_PROMPTS_MAP.keys():
                    suffix = f"_{style}"
                    if name_part.endswith(suffix):
                        prompt_style = style
                        model_name = name_part[: -len(suffix)]
                        break
                if model_name is None or prompt_style is None:

                    parts = name_part.split("_", 1)
                    if len(parts) == 2:
                        model_name, prompt_style = parts[0], parts[1]
                    else:
                        model_name = name_part
                        prompt_style = ""
                model_name = model_name.strip()
                prompt_style = prompt_style.strip()
                if prompt_style not in SYSTEM_PROMPTS_MAP:
                    print(f"Warning: Unrecognized prompt style '{prompt_style}' in file {filename}. Skipping this file.")
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as lf:
                        lecture_text = lf.read().strip()
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}. Skipping this file.")
                    continue

                system_prompt_text = SYSTEM_PROMPTS_MAP.get(prompt_style, "")
                if not system_prompt_text:
                    print(f"Warning: System prompt for style '{prompt_style}' is not defined. Using empty prompt.")

                log_dir = os.path.join("evaluation", "LLM_evaluation_results", mode, doc_id)
                halluc_score, halluc_reason = evaluate_metric("hallucination", system_prompt_text, notes_text, lecture_text)
                help_score, help_reason = evaluate_metric("helpfulness", system_prompt_text, notes_text, lecture_text)
                struct_score, struct_reason = evaluate_metric("structure", system_prompt_text, notes_text, lecture_text)
                tts_score, tts_reason = evaluate_metric("tts", system_prompt_text, notes_text, lecture_text)

                if halluc_score is None:
                    halluc_score, halluc_reason = -1, "Evaluation failed"
                if help_score is None:
                    help_score, help_reason = -1, "Evaluation failed"
                if struct_score is None:
                    struct_score, struct_reason = -1, "Evaluation failed"
                if tts_score is None:
                    tts_score, tts_reason = -1, "Evaluation failed"

                writer.writerow([
                    doc_id,
                    model_name,
                    prompt_style,
                    halluc_score, halluc_reason,
                    help_score, help_reason,
                    struct_score, struct_reason,
                    tts_score, tts_reason
                ])
                f_gt.flush() if mode == "gt" else f_ocr.flush()

                print(f"{mode.upper()} | Doc {doc_id} | Model: {model_name} | Prompt: {prompt_style} -> Evaluated")
                break #remove after testing with one case
            break #remove after testing with one case
