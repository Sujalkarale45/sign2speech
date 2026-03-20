"""
mel_utils.py
Converts audio files to log-mel spectrograms compatible with HiFi-GAN.
Also provides gTTS-based synthesis for glosses without real audio.
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
    """
    Load a .wav file and compute a log-mel spectrogram.

    Returns:
        np.ndarray of shape (80, T), float32.
    """
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
    Synthesize speech from a gloss word via gTTS, then extract log-mel.
    Requires ffmpeg to be installed (available by default on Colab).

    Returns:
        np.ndarray of shape (80, T), float32.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_mp3 = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        
        tts = gTTS(text=gloss.lower(), lang="en", slow=False)
        tts.save(tmp_mp3)
        
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", str(config["sr"]), tmp_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return audio_to_mel(tmp_wav, config)
    
    except FileNotFoundError as e:
        raise RuntimeError(
            f"gTTS synthesis failed for '{gloss}': ffmpeg not found. "
            f"Install with: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)"
        ) from e
    except Exception as e:
        raise RuntimeError(f"gTTS synthesis failed for '{gloss}': {e}") from e
    finally:
        for p in (tmp_mp3, tmp_wav):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass  # Ignore cleanup errors