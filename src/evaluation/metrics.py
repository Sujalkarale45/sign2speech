"""
metrics.py
Evaluation metrics for SignVoice output quality.
  - MCD:  Mel Cepstral Distortion (lower = better, 0 = perfect)
  - MOS proxy: UTMOS-style heuristic from spectral statistics
"""
import numpy as np
import librosa


def mel_cepstral_distortion(ref_wav: str, gen_wav: str,
                             sr: int = 22050, n_mfcc: int = 13) -> float:
    """
    Computes MCD-13 between reference and generated waveforms.

    Args:
        ref_wav: Path to ground-truth .wav.
        gen_wav: Path to generated .wav.
        n_mfcc:  Number of MFCC coefficients (13 is standard).

    Returns:
        MCD value in dB. Typical range: 4–10 dB. <6 dB is good.
    """
    ref, _ = librosa.load(ref_wav, sr=sr)
    gen, _ = librosa.load(gen_wav, sr=sr)

    # Pad/trim to same length
    min_len = min(len(ref), len(gen))
    ref, gen = ref[:min_len], gen[:min_len]

    mfcc_ref = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc)
    mfcc_gen = librosa.feature.mfcc(y=gen, sr=sr, n_mfcc=n_mfcc)

    diff = mfcc_ref - mfcc_gen
    mcd  = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.mean(np.sum(diff**2, axis=0)))
    return float(mcd)


def compute_mos_proxy(gen_wav: str, sr: int = 22050) -> float:
    """
    Heuristic MOS proxy based on spectral flatness and SNR estimate.
    Not a substitute for real MOS but useful for automated evaluation.

    Returns:
        Score in [1.0, 5.0]. >3.5 indicates intelligible speech.
    """
    y, _ = librosa.load(gen_wav, sr=sr)
    flatness  = np.mean(librosa.feature.spectral_flatness(y=y))
    rms       = np.sqrt(np.mean(y**2))
    noise_est = np.percentile(np.abs(y), 5)
    snr       = 20 * np.log10(rms / (noise_est + 1e-8) + 1e-8)
    score     = np.clip(1.0 + (snr / 40.0) * 4.0 * (1.0 - flatness * 10), 1.0, 5.0)
    return float(score)