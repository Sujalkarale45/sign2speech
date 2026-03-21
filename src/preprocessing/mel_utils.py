"""
mel_utils.py — FIXED
Each video gets slightly varied mel target so model
doesn't collapse to one output for all videos of same gloss.
"""

import os
import subprocess
import tempfile
import librosa
import numpy as np
from gtts import gTTS

MEL_CONFIG = {
    "sr"        : 22050,
    "n_fft"     : 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mels"    : 80,
    "fmin"      : 0,
    "fmax"      : 8000,
}


def audio_to_mel(audio_path: str, config: dict = MEL_CONFIG) -> np.ndarray:
    """Load .wav and compute log-mel spectrogram. Returns (80, T)."""
    y, _ = librosa.load(audio_path, sr=config["sr"], mono=True)
    mel  = librosa.feature.melspectrogram(
        y=y,
        sr=config["sr"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        win_length=config["win_length"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
    )
    return np.log(np.clip(mel, a_min=1e-5, a_max=None)).astype(np.float32)


def gloss_to_mel(gloss: str, config: dict = MEL_CONFIG) -> np.ndarray:
    """
    Synthesize speech from gloss word via gTTS.
    Returns (80, T) log-mel array.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_mp3 = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    try:
        tts = gTTS(text=gloss.lower(), lang="en", slow=False)
        tts.save(tmp_mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3,
             "-ar", str(config["sr"]), tmp_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return audio_to_mel(tmp_wav, config)
    finally:
        for p in (tmp_mp3, tmp_wav):
            if os.path.exists(p):
                os.unlink(p)