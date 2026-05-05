import time
from mlx_audio.tts.utils import load_model

model = load_model("mlx-community/Kokoro-82M-bf16")
#model =load_model("mlx-community/orpheus-3b-0.1-ft-6bit")


test_sentences = ['The river ice begins to thaw,',
 'The melting snow is what we saw.',
 'The stars are bright in freezing air,',
 'With quiet magic everywhere.',
 '',
 'Tomorrow brings a Saturday,',
 'A snowy start to find your way.',
 'It will be forty one degrees,',
 'Sleep well beneath the willow trees.']

t0=time.time()
t1=None

for sentence in test_sentences:
    for chunk in model.generate(sentence, voice="af_heart", lang='en'):
        if t1 is None:
            t1=time.time()
        print(chunk)

print(f"Completed generation in: {time.time()-t0}. First chunk in {t1-t0}")
