"""
scripts/infer.py
CLI entrypoint for inference on a single sign video.
Usage:
    python scripts/infer.py --video path/to/sign.mp4 --checkpoint checkpoints/ckpt_epoch030.pt --output out.wav
"""
import argparse, os, sys, yaml, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.signvoice import SignVoiceModel
from src.preprocessing.normalizer import KeypointNormalizer
from src.inference.vocoder import HiFiGANWrapper
from src.inference.pipeline import InferencePipeline


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    normalizer = KeypointNormalizer(config["data"]["stats_path"])
    normalizer.load()

    model = SignVoiceModel(config)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    vocoder  = HiFiGANWrapper(config["vocoder"]["model_path"], device)
    pipeline = InferencePipeline(model, normalizer, vocoder, device)

    out = pipeline.run(args.video, args.output)
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     default="output.wav")
    parser.add_argument("--config",     default="configs/default.yaml")
    main(parser.parse_args())