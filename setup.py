"""
SignVoice: Direct Sign Language Video to Speech synthesis.
Installable package configuration.
"""
from setuptools import setup, find_packages

setup(
    name="signvoice",
    version="0.1.0",
    author="Sujal Karale",
    description="Sign Language Video to Speech — no text modality",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "mediapipe>=0.10.9",
        "opencv-python>=4.9.0",
        "librosa>=0.10.1",
        "gTTS>=2.5.1",
        "numpy>=1.26.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "scipy>=1.12.0",
    ],
)