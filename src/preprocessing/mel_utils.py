"""
mel_utils.py
Utilities for generating log-mel spectrogram targets from glosses.

- Prefers real human-recorded .wav files in data/raw/audio_gloss/
- Falls back to gTTS + slight random variation to avoid mode collapse
- Caches generated TTS audio/mel to avoid redundant synthesis
"""

import os
import subprocess
import tempfile
import warnings
import numpy as np
import librosa
from gtts import gTTS
from pathlib import Path
from typing import Dict, Optional

# ── Configuration ─────────────────────────────────────────────
MEL_CONFIG = {
    "sr": 22050,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mels": 80,
    "fmin": 0,
    "fmax": 8000,
}

# Where real human-recorded audio lives (one file per gloss)
REAL_AUDIO_DIR = Path("data/raw/audio_gloss")

# Cache directory for generated TTS audio & mel (speeds up repeated runs)
CACHE_DIR = Path("data/processed/mel_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Small random variation applied only to synthetic (TTS) mels
VARIATION_STD = 0.04   # quite small — adjust between 0.02–0.08
# ──────────────────────────────────────────────────────────────


def audio_to_mel(audio_path: str | os.PathLike, config: dict = MEL_CONFIG) -> np.ndarray:
    """
    Load .wav file and compute log-mel spectrogram → shape (n_mels, time)
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=config["sr"], mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {audio_path}: {e}")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        win_length=config["win_length"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
        center=True,
        power=2.0,
    )

    log_mel = np.log(np.maximum(mel, 1e-5)).astype(np.float32)
    return log_mel


def _synthesize_tts_to_wav(text: str, lang: str = "en") -> Path:
    """Generate TTS audio with gTTS and save to temporary file."""
    tts = gTTS(text=text, lang=lang, slow=False)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tts.save(tmp_mp3.name)
        tmp_mp3_path = Path(tmp_mp3.name)

    # Convert mp3 → wav using ffmpeg (most reliable cross-platform way)
    wav_path = tmp_mp3_path.with_suffix(".wav")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(tmp_mp3_path),
                "-ar", str(MEL_CONFIG["sr"]),
                "-ac", "1", str(wav_path)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e}")
    finally:
        tmp_mp3_path.unlink(missing_ok=True)

    return wav_path


def _get_cached_tts_mel(gloss: str, config: dict) -> Optional[np.ndarray]:
    """Check if we already have a cached TTS mel for this gloss."""
    cache_path = CACHE_DIR / f"{gloss.lower()}_tts_mel.npy"
    if cache_path.is_file():
        return np.load(cache_path)
    return None


def _cache_tts_mel(gloss: str, mel: np.ndarray):
    """Save generated TTS mel to cache."""
    cache_path = CACHE_DIR / f"{gloss.lower()}_tts_mel.npy"
    np.save(cache_path, mel)


def gloss_to_mel(
    gloss: str,
    config: dict = MEL_CONFIG,
    add_variation: bool = True,
    force_tts: bool = False
) -> np.ndarray:
    """
    Main function: get mel spectrogram for a gloss.

    Priority:
      1. Real human .wav → data/raw/audio_gloss/{gloss}.wav
      2. Cached TTS mel (if previously generated)
      3. Generate TTS → convert → mel → cache → apply small variation

    Args:
        gloss:          The sign gloss / word (e.g. "drink")
        config:         Mel extraction parameters
        add_variation:  Whether to add Gaussian noise (only to TTS)
        force_tts:      Skip real audio and always use TTS (debug)

    Returns:
        np.ndarray: log-mel spectrogram (80, T)
    """
    gloss = gloss.lower().strip()

    # 1. Try real recorded audio (highest quality)
    real_path = REAL_AUDIO_DIR / f"{gloss}.wav"
    if real_path.is_file() and not force_tts:
        try:
            return audio_to_mel(real_path, config)
        except Exception as e:
            warnings.warn(f"Real audio failed for '{gloss}': {e} → falling back to TTS")

    # 2. Check cache for previous TTS result
    cached = _get_cached_tts_mel(gloss, config)
    if cached is not None:
        mel = cached.copy()
    else:
        # 3. Generate TTS
        print(f"  [TTS] Generating audio for gloss: '{gloss}'")
        try:
            tmp_wav = _synthesize_tts_to_wav(gloss)
            mel = audio_to_mel(tmp_wav, config)
            tmp_wav.unlink(missing_ok=True)
            _cache_tts_mel(gloss, mel)
        except Exception as e:
            raise RuntimeError(f"TTS generation failed for '{gloss}': {e}")

    # Add small random variation only to synthetic targets
    if add_variation:
        noise = np.random.normal(0, VARIATION_STD, mel.shape).astype(np.float32)
        mel = mel + noise

    return mel


# For backward compatibility / testing
def get_mel_spectrogram(gloss: str, **kwargs) -> np.ndarray:
    """Alias for gloss_to_mel — used in some older code."""
    return gloss_to_mel(gloss, **kwargs)


if __name__ == "__main__":
    # Quick test
    test_glosses = ["drink", "hello", "mother", "not", "bad"]
    for g in test_glosses:
        try:
            m = gloss_to_mel(g)
            print(f"{g:12} → shape {m.shape} | mean {m.mean():.4f} | std {m.std():.4f}")
        except Exception as e:
            print(f"Failed {g}: {e}")