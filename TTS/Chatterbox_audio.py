import torchaudio
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = (
    "Welcome to this introductory lecture on machine learning. "
    "Today, we'll cover key concepts such as supervised learning, model training, and evaluation techniques."
)

audio = model.generate(text)

torchaudio.save("TTS_outputs/test_chatterbox.wav", audio, model.sr)
print("Saved: TTS_outputs/test_chatterbox.wav")
