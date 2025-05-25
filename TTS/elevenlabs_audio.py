import os
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

lecture_text = """
Welcome to today's class on probability theory. <break time='1s'/>
Let's begin with a simple idea: <prosody rate='slow'>What is the chance of flipping a coin and getting heads?</prosody> <break time='1s'/>
<emphasis level='moderate'>It's fifty percent</emphasis> — assuming the coin isn't rigged! <prosody pitch='+10%'>Sounds easy, right?</prosody> <break time='1s'/>
But what if I told you that understanding coin flips helps explain things like <emphasis level='moderate'>genetics</emphasis>, <emphasis level='moderate'>machine learning</emphasis>, and even <emphasis level='strong'>financial risk?</emphasis> <break time='1s'/>
Get ready — because probability is everywhere.
"""

voice_id = "EXAVITQu4vr4xnSDxMaL"

audio_stream = client.text_to_speech.convert(
    text=lecture_text,
    voice_id=voice_id,
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128"
)  

os.makedirs("TTS_outputs", exist_ok=True)
output_path = "TTS_outputs/elevenlabs_sample_lecture.mp3"

with open(output_path, "wb") as f:
    for chunk in audio_stream:
        if chunk:
            f.write(chunk)

print(f"✅ Audio saved as {output_path}")
