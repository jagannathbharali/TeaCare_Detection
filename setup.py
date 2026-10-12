#!/usr/bin/env python3
"""
setup.py
────────
One-time setup script: creates folders, downloads YOLOv8 base weights,
and checks Ollama connectivity.

Run: python setup.py
"""

import sys
import subprocess
from pathlib import Path


def check_python():
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 10:
        print(f"✗ Python 3.10+ required (you have {major}.{minor})")
        sys.exit(1)
    print(f"✓ Python {major}.{minor}")


def create_dirs():
    dirs = [
        "data/train", "data/val", "data/test",
        "models", "reports", "assets",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directory structure created")


def install_deps():
    print("\nInstalling Python dependencies ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
        "--quiet"
    ])
    print("✓ Dependencies installed")


def download_yolo_weights():
    print("\nDownloading YOLOv8m-cls base weights ...")
    try:
        from ultralytics import YOLO
        YOLO("yolov8m-cls.pt")   # auto-downloads to ~/.ultralytics/
        print("✓ YOLOv8 weights ready")
    except Exception as e:
        print(f"⚠ Could not download weights: {e}")
        print("  They will auto-download on first training run.")


def check_ollama():
    print("\nChecking Ollama ...")
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        if "mistral" in " ".join(models):
            print("✓ Ollama running, Mistral available")
        else:
            print("⚠ Ollama running but Mistral not pulled.")
            print("  Run: ollama pull mistral")
    except Exception:
        print("⚠ Ollama not reachable.")
        print("  Install from https://ollama.ai then run:")
        print("    ollama serve")
        print("    ollama pull mistral")


def print_next_steps():
    print("""
╔══════════════════════════════════════════════════════╗
║              TeaGuard AI — Setup Complete            ║
╚══════════════════════════════════════════════════════╝

Next steps:

  1. Place your Tea_Dataset folder next to this project:
     tea_disease_app/
     Tea_Dataset/
         ├── train/<ClassName>/
         ├── test/<ClassName>/
         └── valid/<ClassName>/

  2. Train the model:
     python train_model.py

     Or with explicit path:
     python train_model.py --data /path/to/Tea_Dataset

  3. Launch the desktop app:
     python main.py

  4. For AI economic reports, ensure Ollama + Mistral:
     ollama serve
     ollama pull mistral
""")


if __name__ == "__main__":
    print("TeaGuard AI — Setup\n" + "="*40)
    check_python()
    create_dirs()
    install_deps()
    download_yolo_weights()
    check_ollama()
    print_next_steps()
