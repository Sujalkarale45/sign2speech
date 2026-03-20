"""
pipeline.py
End-to-end inference: sign video → keypoints → mel → waveform → .wav file.
"""
import numpy as np
import torch
import soundfile as sf
from src.preprocessing.extractor import KeypointExtractor
from src.preprocessing.normalizer import KeypointNormalizer
from .vocoder import HiFiGANWrapper


class InferencePipeline:
    """
    Wraps the full SignVoice inference graph.
    Usage:
        pipe = InferencePipeline(model, normalizer, vocoder, device="cuda")
        pipe.run("sign_video.mp4", "output.wav")
    """

    def __init__(self, model, normalizer: KeypointNormalizer,
                 vocoder: HiFiGANWrapper, device: str = "cuda",
                 max_mel_frames: int = 1000):
        self.model          = model.eval().to(device)
        self.normalizer     = normalizer
        self.vocoder        = vocoder
        self.device         = device
        self.max_mel_frames = max_mel_frames
        self.extractor      = KeypointExtractor()

    @torch.no_grad()
    def run(self, video_path: str, output_wav: str, sr: int = 22050) -> str:
        """
        Full pipeline: video → .wav.

        Args:
            video_path: Path to input sign language video.
            output_wav: Destination .wav path.
            sr:         Sample rate (must match HiFi-GAN training).

        Returns:
            output_wav path.
        """
        # 1. Extract + normalize keypoints
        kp  = self.extractor.process_video(video_path)           # (T, 183)
        kp  = self.normalizer.normalize(kp)                      # (T, 183)
        kp_t = torch.from_numpy(kp).unsqueeze(0).to(self.device) # (1, T, 183)
        key_lens = torch.tensor([kp_t.size(1)], device=self.device)

        # 2. Autoregressive mel decoding
        mel_frame = torch.zeros(1, 1, 80, device=self.device)    # <SOS>
        mel_frames = []

        for _ in range(self.max_mel_frames):
            enc_out = self.model.encoder(kp_t, src_key_padding_mask=None)
            mel_pre, stop_logit = self.model.decoder(enc_out, mel_frame)
            next_frame = mel_pre[:, -1:, :]                      # (1, 1, 80)
            mel_frames.append(next_frame)
            mel_frame = torch.cat([mel_frame, next_frame], dim=1)
            if torch.sigmoid(stop_logit[:, -1, 0]) > 0.5:
                break

        mel = torch.cat(mel_frames, dim=1).squeeze(0).T          # (80, T_mel)

        # 3. PostNet refinement
        mel_post = self.model.postnet(mel.unsqueeze(0)).squeeze(0)  # (80, T_mel)

        # 4. Vocoder → waveform
        wav = self.vocoder.mel_to_wav(mel_post.unsqueeze(0))     # (1, N)
        sf.write(output_wav, wav.squeeze().cpu().numpy(), sr)
        return output_wav