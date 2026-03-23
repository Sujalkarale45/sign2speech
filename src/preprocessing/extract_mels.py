"""
extract_mels.py
Converts every clipped .wav file into an (80, T) log-mel spectrogram
and saves it as a .npy file.

Expected input:
  data/processed/audio_clips/<SENTENCE_NAME>.wav

Output:
  data/processed/mels/<SENTENCE_NAME>.npy
"""

import librosa
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
AUDIO_CLIPS_DIR = Path("data/processed/audio_clips")
OUT_DIR         = Path("data/processed/mels")
SR              = 22050
N_MELS          = 80
HOP_LENGTH      = 256
WIN_LENGTH      = 1024
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def extract_mel(wav_path: Path) -> np.ndarray:
    y, _  = librosa.load(str(wav_path), sr=SR)
    mel   = librosa.feature.melspectrogram(
                y=y, sr=SR,
                n_mels=N_MELS,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH)
    return librosa.power_to_db(mel, ref=np.max)   # (80, T)


if __name__ == "__main__":
    wavs = sorted(AUDIO_CLIPS_DIR.glob("*.wav"))
    print(f"Audio clips found: {len(wavs)}")
    ok = skip = err = 0

    for wav in wavs:
        out = OUT_DIR / f"{wav.stem}.npy"
        if out.exists():
            skip += 1
            continue
        try:
            mel = extract_mel(wav)
            np.save(str(out), mel)
            print(f"  [OK]   {wav.stem}  shape={mel.shape}")
            ok += 1
        except Exception as exc:
            print(f"  [ERR]  {wav.stem}: {exc}")
            err += 1

    print(f"\nDone → {ok} saved, {skip} skipped, {err} errors")
    print(f"Mels saved to: {OUT_DIR}")
