import os, csv
from datetime import datetime

os.environ.setdefault("OPENAI_API_KEY", "ollama")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:22134/v1")

from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

def to_int_1to10(score01: float) -> int:
    try:
        if score01 is None: return -1
        s = float(score01)
        if s == 0: return 0
        v = int(round(s * 10))
        return max(1, min(v, 10))
    except:
        return -1

def main():
    system_prompt = (
        "You are a knowledgeable and engaging professor for spoken delivery. "
        "Use simple explanations, clear punctuation, and conversational tone."
    )
    notes = (
        "GDP measures the total value of goods and services produced domestically. "
        "High GDP growth attracts foreign investment; low or negative growth suggests caution."
    )
    lecture = (
        "Gross Domestic Product (GDP) reflects a nation's overall production. "
        "When GDP growth is strong, companies often expand; when it slows or turns negative, "
        "businesses become more cautious about investment."
    )

    helpfulness = GEval(
        name="Helpfulness",
        criteria=("Evaluate how helpful and educational the lecture is for a business student, "
                  "given the system prompt and notes. Consider coverage of key points and clarity."),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    structure = GEval(
        name="StructureClarity",
        criteria=("Evaluate structure and clarity for spoken delivery. Consider logical flow, "
                  "signposting/sections, transitions, and ease of following explanations."),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    tts_compliance = GEval(
        name="TTSCompliance",
        criteria=("Evaluate compliance with the system prompt's style/format rules for TTS delivery. "
                  "Check adherence to tone/style guidance and avoiding disallowed content/metadata."),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    faithfulness = FaithfulnessMetric()  

    composite_input = (
        "SYSTEM PROMPT (style/tts):\n" + system_prompt + "\n\n"
        "NOTES (source content):\n" + notes
    )
    tc = LLMTestCase(
        input=composite_input,
        actual_output=lecture,
        retrieval_context=[notes],
    )

    helpfulness.measure(tc)
    structure.measure(tc)
    tts_compliance.measure(tc)
    faithfulness.measure(tc)

    h_score, h_reason = to_int_1to10(helpfulness.score), (helpfulness.reason or "").strip()
    s_score, s_reason = to_int_1to10(structure.score), (structure.reason or "").strip()
    t_score, t_reason = to_int_1to10(tts_compliance.score), (tts_compliance.reason or "").strip()
    f_score, f_reason = to_int_1to10(faithfulness.score), (faithfulness.reason or "").strip()

    print("\nDeepEval backend: Ollama (configured via `deepeval set-ollama`)")
    print("Scores (1–10):")
    print(f"  Helpfulness               : {h_score} | {h_reason}")
    print(f"  Structure & Clarity       : {s_score} | {s_reason}")
    print(f"  TTS Prompt Compliance     : {t_score} | {t_reason}")
    print(f"  Hallucination/Faithfulness: {f_score} | {f_reason}")

    out_csv = f"deepeval_smoketest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "document_id","model","prompt_style",
            "hallucination_score","hallucination_reason",
            "helpfulness_score","helpfulness_reason",
            "structure_score","structure_reason",
            "tts_score","tts_reason"
        ])
        w.writerow([
            "smoketest-doc","(deepeval set-ollama default)","(N/A)",
            f_score, f_reason,
            h_score, h_reason,
            s_score, s_reason,
            t_score, t_reason
        ])
    print(f"\nSaved CSV: {out_csv}")

if __name__ == "__main__":
    main()