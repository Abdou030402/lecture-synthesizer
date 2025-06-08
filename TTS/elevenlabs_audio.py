import os
from elevenlabs.client import ElevenLabs


def synthesize_audio(text: str, voice_id: str, output_path: str,
                     model_id: str = "eleven_multilingual_v2") -> None:
    """Convert ``text`` to speech using the specified ElevenLabs voice.

    Parameters
    ----------
    text: str
        The text to synthesize. Can contain SSML markup.
    voice_id: str
        The ElevenLabs voice ID to use.
    output_path: str
        File path where the resulting MP3 will be written.
    model_id: str, optional
        ElevenLabs model identifier. Defaults to ``eleven_multilingual_v2``.
    """

    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format="mp3_44100_128",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)
    print(f"\u2705 Audio saved as {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate speech audio from a text file or directory of texts"
    )
    parser.add_argument(
        "input",
        help="Path to a .txt file or a directory containing text files",
    )
    parser.add_argument(
        "--voice",
        default="EXAVITQu4vr4xnSDxMaL",
        help="ElevenLabs voice ID",
    )
    parser.add_argument(
        "--model",
        default="eleven_multilingual_v2",
        help="ElevenLabs model ID",
    )
    parser.add_argument(
        "--output-dir",
        default="TTS_outputs",
        help="Directory where MP3 files will be written",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    def process_file(path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output_dir, f"{base}.mp3")
        synthesize_audio(text, args.voice, out_path, model_id=args.model)

    if os.path.isdir(args.input):
        for name in sorted(os.listdir(args.input)):
            if name.lower().endswith(".txt"):
                process_file(os.path.join(args.input, name))
    else:
        process_file(args.input)
