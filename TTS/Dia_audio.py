import soundfile as sf
from dia.model import Dia

model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

text = (
    "[S1] Today, we're going to explore the fundamentals of neural networks. (clears throat) "
    "We'll look at how individual neurons operate and how they're connected to form complex learning systems. "
    "By the end of this lecture, you should understand the basic structure of a feedforward neural network. (smiles)"
)

audio = model.generate(text)
sf.write("lecture_dia.wav", audio, 44100)
print("Saved: lecture_dia.wav")
