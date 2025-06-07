import os
from elevenlabs.client import ElevenLabs

lecture_text_example = """
Welcome to today's class on probability theory. <break time='1s'/>
Let's begin with a simple idea: <prosody rate='slow'>What is the chance of flipping a coin and getting heads?</prosody> <break time='1s'/>
<emphasis level='moderate'>It's fifty percent</emphasis> — assuming the coin isn't rigged! <prosody pitch='+10%'>Sounds easy, right?</prosody> <break time='1s'/>
But what if I told you that understanding coin flips helps explain things like <emphasis level='moderate'>genetics</emphasis>, <emphasis level='moderate'>machine learning</emphasis>, and even <emphasis level='strong'>financial risk?</emphasis> <break time='1s'/>
Get ready — because probability is everywhere.
"""

def synthesize_audio(text: str, voice_id: str, output_path: str,
                     model_id: str = "eleven_multilingual_v2") -> None:
    
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
    print(f"✅ Audio saved as {output_path}")
 

if __name__ == "__main__":
    input_txt_path = "nlp_outputs/trocr_output_handwritten6_llama3_output_2.txt" 
    voice_id = "EXAVITQu4vr4xnSDxMaL"

    if not os.path.isfile(input_txt_path):
        raise FileNotFoundError(f"Input file not found: {input_txt_path}")

    with open(input_txt_path, "r", encoding="utf-8") as file:
        input_text = file.read().strip()

    base_name = os.path.splitext(os.path.basename(input_txt_path))[0]
    output_mp3_path = f"TTS_outputs/{base_name}.mp3"

    synthesize_audio(input_text, voice_id, output_mp3_path)
