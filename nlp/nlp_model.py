import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """
You are a knowledgeable and engaging university professor delivering a spoken lecture based on a set of handwritten or scanned notes.

Your job is to:
- Expand and clarify the notes into full, natural-sounding spoken paragraphs.
- Simplify complex concepts so that a beginner-level student can easily understand them.
- Use analogies, simple examples, and explanations where helpful.
- Make the lecture flow smoothly, as if you were speaking in a real classroom.
- Add tone and speech direction using square brackets for emotional guidance to a TTS system. Examples include: [serious], [enthusiastic], [pause], [curious], [emphasize], [reflective].

Your final output should only be the professor-style spoken content do not include any extra comments, headings, or metadata.
"""

def generate_professor_lecture(notes: str, model: str) -> str:
    """
    Generate professor-style lecture from notes using specified model via Ollama.
    """
    payload = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\nLecture Notes:\n{notes}\n\nLecture Script:",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.RequestException as e:
        return f"[ERROR] Failed with model '{model}': {e}"


def chunk_text(text: str, max_words: int = 600) -> list:
    """
    Split text into chunks of approximately max_words words for token-limited models.
    """
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def combine_chunks(enhanced_chunks: list) -> str:
    """
    Combine enhanced text chunks with logical spoken transitions.
    """
    result = []
    for i, chunk in enumerate(enhanced_chunks):
        if i > 0:
            result.append("[pause] Let's move on to the next part of the lecture.")
        result.append(chunk)
    return "\n\n".join(result)
