import requests
from system_prompts import SYSTEM_PROMPTS_MAP

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_professor_lecture(notes: str, ollama_model_name: str, system_prompt_type: str) -> str:
    system_prompt = SYSTEM_PROMPTS_MAP.get(system_prompt_type)

    if not system_prompt:
        return f"[ERROR] Invalid system_prompt_type: '{system_prompt_type}'. Available types: {list(SYSTEM_PROMPTS_MAP.keys())}"

    payload = {
        "model": ollama_model_name,
        "prompt": f"{system_prompt}\n\nLecture Notes:\n{notes}\n\nLecture Script:",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return f"[ERROR] Could not connect to Ollama at {OLLAMA_URL}. Is Ollama running?"
    except requests.RequestException as e:
        return f"[ERROR] Failed with model '{ollama_model_name}' and prompt type '{system_prompt_type}': {e}"
    except KeyError:
        return f"[ERROR] Unexpected response format from Ollama. Response: {response.text}"


def chunk_text(text: str, max_words: int = 600) -> list:
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def combine_chunks(enhanced_chunks: list) -> str:
    result = []
    for i, chunk in enumerate(enhanced_chunks):
        if i > 0:
            result.append("[pause] Let's move on to the next part of the lecture.")
        result.append(chunk)
    return "\n\n".join(result)
