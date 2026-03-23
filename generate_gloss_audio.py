from gtts import gTTS
from pathlib import Path
from pydub import AudioSegment
import os
import random

# Your signs from the project
TARGET_SIGNS = [
    'drink', 'who', 'cow', 'bird', 'brown',
    'cat', 'kiss', 'go', 'think', 'man'
]

OUTPUT_DIR = Path("data/raw/audio_gloss")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Variations to make audio less repetitive
SPEEDS = [0.85, 0.92, 1.0, 1.08, 1.15]
ACCENTS = ["com", "co.uk", "ca", "co.in", "com.au"]  # different English regions

def create_variant(word: str, speed: float, accent: str, variant_num: int):
    file_path = OUTPUT_DIR / f"{word}_{variant_num}.wav"

    if file_path.exists():
        print(f"Already exists → {file_path}")
        return

    print(f"Creating {file_path} (speed={speed:.2f}, accent={accent})")

    # Generate TTS
    tts = gTTS(text=word, lang='en', tld=accent, slow=False)
    temp_mp3 = f"temp_{word}.mp3"
    tts.save(temp_mp3)

    # Load & modify speed → save as wav 22050 Hz (matches your mel config)
    audio = AudioSegment.from_mp3(temp_mp3)
    # pydub speed change
    if speed != 1.0:
        audio = audio.speedup(playback_speed=speed) if speed > 1 else \
                audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed)})

    audio = audio.set_frame_rate(22050).set_channels(1)
    audio.export(file_path, format="wav")

    os.remove(temp_mp3)  # cleanup
    print(f"   → Done: {file_path}")

# Generate 4 variants per word
for word in TARGET_SIGNS:
    print(f"\n=== {word.upper()} ===")
    random.shuffle(SPEEDS)
    random.shuffle(ACCENTS)

    for i in range(4):
        create_variant(word, SPEEDS[i % len(SPEEDS)], ACCENTS[i % len(ACCENTS)], i+1)

print("\nFinished! Folder created:", OUTPUT_DIR.absolute())
print("Now run: python scripts/preprocess_asl.py")