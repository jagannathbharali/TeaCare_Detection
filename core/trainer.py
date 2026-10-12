
from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml
from loguru import logger
from ultralytics import YOLO

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _load_cfg() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

#  data.yaml loader & path fixer

def load_roboflow_yaml(data_yaml_path: str | Path) -> dict:

    path = Path(data_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"data.yaml not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    logger.info(f"Loaded data.yaml: nc={data.get('nc')}, names={data.get('names')}")
    return data


def fix_data_yaml_paths(data_yaml_path: str | Path) -> Path:

    orig_path    = Path(data_yaml_path).resolve()
    dataset_root = orig_path.parent

    with open(orig_path) as f:
        data = yaml.safe_load(f)

    split_dirs = {}
    for split_key in ["train", "val", "test"]:
        raw = data.get(split_key, "")
        if not raw:
            continue
        raw_path = Path(raw)
        parts    = raw_path.parts
        try:
            idx = next(
                i for i in range(len(parts))
                if Path(*parts[i:]).is_relative_to(dataset_root)
                or parts[i] in ("train", "valid", "test")
            )
            relative = str(Path(*parts[idx:]))
        except (StopIteration, TypeError):
            relative = str(Path(*parts[-2:]))

        split_dirs[split_key] = relative
        logger.debug(f"  {split_key}: {raw} → {relative}")

    fixed = dict(data)
    for k, v in split_dirs.items():
        fixed[k] = v

    fixed_path = dataset_root / "data_fixed.yaml"
    with open(fixed_path, "w") as f:
        yaml.dump(fixed, f, default_flow_style=False)

    logger.info(f"Fixed data.yaml written to: {fixed_path}")
    return fixed_path


def find_data_yaml(dataset_root: str | Path) -> Path:

    root = Path(dataset_root).resolve()
    candidates = [
        root / "data.yaml",
        root / "data_fixed.yaml",
    ] + list(root.glob("*/data.yaml"))

    for c in candidates:
        if c.exists():
            logger.info(f"Found data.yaml: {c}")
            return c

    raise FileNotFoundError(
        f"No data.yaml found in {root}.\n"
        f"Contents: {[p.name for p in root.iterdir()]}"
    )

#  Dataset validation

def validate_dataset(dataset_root: str | Path) -> bool:

    root = Path(dataset_root).resolve()

    try:
        yaml_path = find_data_yaml(root)
        data      = load_roboflow_yaml(yaml_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return False

    logger.info(f"Dataset root : {root}")
    logger.info(f"Classes ({data['nc']}): {data['names']}")

    ok = True
    for split in ["train", "val", "test"]:
        raw = data.get(split, "")
        if not raw:
            logger.warning(f"  {split}: not defined in data.yaml")
            continue

        img_dir = Path(raw)
        if not img_dir.exists():
            img_dir = root / raw
        if not img_dir.exists():
            candidates = list(root.rglob(f"{split}/images"))
            img_dir = candidates[0] if candidates else img_dir

        lbl_dir = img_dir.parent / "labels"

        n_imgs = len([f for f in img_dir.glob("*")
                      if f.suffix.lower() in _IMG_EXTS]) \
                 if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*.txt"))) \
                 if lbl_dir.exists() else 0

        status = "✓" if n_imgs > 0 else "✗"
        logger.info(f"  {status} {split}: {n_imgs} images, {n_lbls} labels")

        if n_imgs == 0:
            logger.warning(f"  No images found in {img_dir}")
            ok = False

    return ok


def sync_classes_to_config(names: list[str] | dict, nc: int):
    """Write discovered class names back to config.yaml."""
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # 🚨 THE FIX: If 'names' is a dictionary, extract just the string values
    if isinstance(names, dict):
        names = list(names.values())

    cfg["diseases"]["classes"] = names
    for cls in names:
        if cls not in cfg["diseases"]["severity"]:
            cfg["diseases"]["severity"][cls] = 1
        if cls not in cfg["diseases"]["crop_loss_pct"]:
            cfg["diseases"]["crop_loss_pct"][cls] = 20.0

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    from loguru import logger
    logger.info(f"config.yaml synced: {nc} classes — {names}")
#  Training

def train_model(
    dataset_root: str | Path | None = None,
    *,
    progress_callback=None,
) -> Path:
    cfg       = _load_cfg()
    ycfg      = cfg["yolo"]
    paths_cfg = cfg["paths"]

    dataset_root = Path(dataset_root or paths_cfg["data_root"]).resolve()
    model_dir    = Path(paths_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Find & fix data.yaml ──────────────────────────────────────────
    logger.info(f"Dataset root: {dataset_root}")
    yaml_path  = find_data_yaml(dataset_root)
    data       = load_roboflow_yaml(yaml_path)

    # Fix paths so they work on any machine
    fixed_yaml = fix_data_yaml_paths(yaml_path)
    data       = load_roboflow_yaml(fixed_yaml)

    # Sync class names to config
    sync_classes_to_config(data["names"], data["nc"])

    # ── Load YOLO model ───────────────────────────────────────────────
    model_base = ycfg.get("model_base", "yolov8m.pt")
    logger.info(f"Base model: {model_base}")
    model = YOLO(model_base)

    # ── Device ────────────────────────────────────────────────────────
    import torch
    device = ("0" if torch.cuda.is_available() else "cpu") \
             if ycfg["device"] == "auto" else ycfg["device"]
    logger.info(f"Device: {device}")

    # ── Epoch callback ────────────────────────────────────────────────
    if progress_callback:
        def on_epoch_end(trainer):
            metrics = getattr(trainer, "metrics", {}) or {}
            progress_callback(trainer.epoch + 1, trainer.epochs, dict(metrics))
        model.add_callback("on_train_epoch_end", on_epoch_end)

    # ── Train ─────────────────────────────────────────────────────────
    logger.info(f"Training with data.yaml: {fixed_yaml}")
    model.train(
        data=str(fixed_yaml),
        task=ycfg.get("task", "detect"),
        imgsz=ycfg["imgsz"],
        epochs=ycfg["epochs"],
        batch=ycfg["batch"],
        lr0=ycfg["lr0"],
        patience=ycfg["patience"],
        workers=ycfg["workers"],
        device=device,
        augment=ycfg["augment"],
        project=str(model_dir),
        name="tea_disease_det",
        exist_ok=True,
        verbose=True,
    )

    # ── Find weights ──────────────────────────────────────────────────
    best = model_dir / "tea_disease_det" / "weights" / "best.pt"
    if not best.exists():
        pts = list(model_dir.rglob("best.pt"))
        if pts:
            best = sorted(pts)[-1]
        else:
            raise FileNotFoundError("Training complete but best.pt not found.")

    logger.success(f"Done → {best}")
    return best

#  Evaluation

def evaluate_model(weights_path: str | Path) -> dict:
    """Run on test split and return mAP metrics."""
    cfg       = _load_cfg()
    root      = Path(cfg["paths"]["data_root"]).resolve()
    yaml_path = find_data_yaml(root)
    fixed     = fix_data_yaml_paths(yaml_path)

    model   = YOLO(str(weights_path))
    metrics = model.val(
        data=str(fixed),
        split="test",
        imgsz=cfg["yolo"]["imgsz"],
    )

    results = {
        "mAP50":     round(float(metrics.box.map50), 4),
        "mAP50-95":  round(float(metrics.box.map),   4),
        "precision": round(float(metrics.box.mp),    4),
        "recall":    round(float(metrics.box.mr),    4),
    }
    logger.info(f"Test metrics: {results}")
    return results
