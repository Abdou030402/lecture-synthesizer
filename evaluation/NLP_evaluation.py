import os
import csv
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import HallucinationMetric, GEval
from deepeval.models import OpenAILLM

NOTES_FOLDER = "notes"
LECTURES_FOLDER = "lectures"
OUTPUT_CSV = "evaluation_results.csv"
GPT_MODEL = "gpt-4o"

judge_model = OpenAILLM(model=GPT_MODEL)

hallucination_metric = HallucinationMetric(model=judge_model)
helpfulness_metric = GEval(
    name="Helpfulness",
    criteria="Does the lecture help explain the input notes to a student?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=judge_model
)

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

notes_files = sorted([f for f in os.listdir(NOTES_FOLDER) if f.endswith(".txt")])
lecture_files = sorted([f for f in os.listdir(LECTURES_FOLDER) if f.endswith(".txt")])

assert len(notes_files) == len(lecture_files), "Mismatch in number of files!"

results = []

for note_file, lecture_file in zip(notes_files, lecture_files):
    notes_path = os.path.join(NOTES_FOLDER, note_file)
    lecture_path = os.path.join(LECTURES_FOLDER, lecture_file)

    notes = read_file(notes_path)
    lecture = read_file(lecture_path)

    test_case = LLMTestCase(
        input=notes,
        actual_output=lecture,
        context=[notes]
    )

    hallucination_metric.measure([test_case])
    helpfulness_metric.measure([test_case])

    results.append({
        "note_file": note_file,
        "lecture_file": lecture_file,
        "hallucination_score": round(hallucination_metric.score, 3),
        "hallucination_reason": hallucination_metric.reason,
        "helpfulness_score": round(helpfulness_metric.score, 3),
        "helpfulness_reason": helpfulness_metric.reason,
    })

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Evaluation complete. Results saved to: {OUTPUT_CSV}")
