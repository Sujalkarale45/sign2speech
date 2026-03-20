# SignVoice

## Overview

SignVoice is a PyTorch-based research project for converting sign language videos to speech. It uses deep learning models to extract keypoints from videos, normalize them, and generate mel spectrograms which are then converted to audio using a vocoder.

## Architecture

[Architecture diagram placeholder]

The system consists of:
- Preprocessing: Keypoint extraction using MediaPipe, normalization, and mel spectrogram utilities.
- Dataset: Custom PyTorch dataset with collation for variable-length sequences.
- Models: Transformer-based encoder, Tacotron-style decoder with cross-attention, PostNet for refinement, optional emotion embedding, and the full SignVoice model.
- Training: Trainer class, custom losses, and learning rate scheduler.
- Inference: Pipeline for video to audio conversion using HiFi-GAN vocoder.
- Evaluation: Metrics for mel cepstral distortion and MOS proxy.

## Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download required datasets and place in `data/raw/`.
4. Run preprocessing: `python scripts/preprocess.py --video_dir data/raw/`

## Usage

- Train the model: `python scripts/train.py --config configs/default.yaml`
- Run inference: `python scripts/infer.py --video path/to/video.mp4`

## Dataset

The dataset should consist of sign language videos paired with corresponding audio or text glosses. Videos are processed to extract keypoints, and audio is converted to mel spectrograms. Place raw videos in `data/raw/` and run the preprocessing script.