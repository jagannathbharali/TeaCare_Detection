# 🍃 TeaGuard AI

**Tea Leaf Disease Detection & Economic Impact Predictor**

A production-ready desktop application that combines YOLOv8 computer vision
with a local Mistral LLM to detect tea leaf diseases and generate detailed
economic impact reports — all running locally on your machine.

---

## Features

| Feature | Technology |
|---|---|
| Image restoration pipeline | NLMeans · CLAHE · Butterworth · Unsharp mask |
| Disease detection | YOLOv8m classification |
| 8 disease classes | Blister Blight, Brown Blight, Grey Blight, Red Leaf Spot, etc. |
| Economic analysis | Local Mistral LLM via Ollama |
| PDF export | ReportLab + matplotlib charts |
| Desktop UI | PyQt5 — dark botanical theme |

---

## Quick Start

```bash
# 1. Clone / download the project
cd tea_disease_app

# 2. Run setup (installs deps, creates folders, checks Ollama)
python setup.py

# 3. Place Tea_Dataset next to tea_disease_app/ (train/test/valid splits)
#    Class names are auto-discovered — no manual config needed

# 4. Train the model
python train_model.py
# or with explicit path:
python train_model.py --data ../Tea_Dataset

# 5. Launch the app
python main.py
```

---

## Dataset Structure

Your `Tea_Dataset` folder should sit **next to** the `tea_disease_app/` folder:

```
your_project/
├── tea_disease_app/      ← this app
└── Tea_Dataset/          ← your dataset
    ├── train/
    │   ├── Healthy/
    │   ├── Blister Blight/
    │   ├── Brown Blight/
    │   └── ...
    ├── test/
    │   └── (same class structure)
    └── valid/            ← note: 'valid' not 'val' — handled automatically
        └── (same class structure)
```

**Class names are auto-discovered** from the `train/` sub-folder names at runtime.
You do not need to edit `config.yaml` manually — the app reads whatever folders exist.

---

## Project Structure

```
tea_disease_app/
├── main.py                  # Desktop app entry point
├── train_model.py           # Standalone training script
├── setup.py                 # One-time setup
├── config.yaml              # All configuration
├── requirements.txt
│
├── core/
│   ├── preprocessing.py     # Image restoration pipeline
│   ├── detector.py          # YOLOv8 inference engine
│   ├── trainer.py           # YOLOv8 training
│   └── economics.py         # Economic calculator + LLM report
│
├── ui/
│   ├── main_window.py       # Full PyQt5 application
│   ├── widgets.py           # Reusable UI components
│   └── styles.py            # Design system (colours, fonts)
│
├── utils/
│   └── report_exporter.py   # PDF + JSON export
│
├── data/                    # Your dataset goes here
├── models/                  # Trained weights saved here
└── reports/                 # Exported reports saved here
```

---

## Preprocessing Pipeline

Each uploaded image goes through 5 restoration stages:

```
Original → NL-Means Denoise → CLAHE (LAB) → Butterworth LP Filter
         → Unsharp Mask → Resize 224×224 → Model inference
```

Toggle preprocessing on/off in the Detect tab to compare raw vs restored results.

---

## Ollama Setup (for economic reports)

```bash
# Install Ollama from https://ollama.ai

# Start server
ollama serve

# Download Mistral
ollama pull mistral

# Verify
curl http://localhost:11434/api/tags
```

The economic report works offline with a template fallback even if Ollama is unavailable.

---

## Configuration

Edit `config.yaml` to customise:
- Disease class names and severity weights
- YOLO training hyperparameters
- Preprocessing filter strengths
- Economic baseline values (yield, price per kg, estate size)
- Ollama host and model name

---

## Recommended Datasets

- **PlantVillage** (via Kaggle) — general plant disease dataset
- **Tea Leaf Disease Dataset** (Kaggle: shashwatwork/tea-leaf-disease-dataset)
- Your own field images organised into class folders

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB |
| GPU | CPU-only | NVIDIA GPU (CUDA) |
| Storage | 5 GB | 20 GB |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Any |
