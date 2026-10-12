from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from ultralytics import YOLO

from core.preprocessing import preprocess_image


def _load_cfg() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


CFG = _load_cfg()


def _display_name(raw: str) -> str:
    """Fuzzy maps raw dataset names to clean UI display names."""
    r = str(raw).strip().lower()
    if "black rot" in r: return "Black Rot"
    if "brown blight" in r: return "Brown Blight"
    if "leaf rust" in r: return "Leaf Rust"
    if "red spider" in r: return "Red Spider Attack"
    if "mosquito bug" in r: return "Mosquito Bug"
    if "white spot" in r: return "White Spot"
    if "healthy" in r or "tea leaf" in r: return "Healthy"
    return "Generic Infection"

# --- Data classes ---

@dataclass
class Detection:
    """Single bounding-box detection within one image."""
    class_id: int
    class_name: str
    display_name: str
    confidence: float
    bbox_xyxy: list[float]  # [x1, y1, x2, y2] in pixels


@dataclass
class SingleResult:
    """All detections for one image."""
    image_path: str
    detections: list[Detection] = field(default_factory=list)
    preprocessed_img: Optional[np.ndarray] = field(default=None, repr=False)
    original_img: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def top_detection(self) -> Optional[Detection]:
        if not self.detections: return None
        return max(self.detections, key=lambda d: d.confidence)

    @property
    def top_class_raw(self) -> str:
        top = self.top_detection
        return top.class_name if top else "Tea leaf"

    @property
    def top_disease(self) -> str:
        top = self.top_detection
        return top.display_name if top else "No Detection"

    @property
    def confidence(self) -> float:
        top = self.top_detection
        return top.confidence if top else 0.0

    @property
    def is_diseased(self) -> bool:
        # Use the standardized display name to avoid substring collisions
        disp = self.top_disease.strip().lower()
        return disp not in ["healthy", "no detection"]

    @property
    def crop_loss_pct(self) -> float:
        raw = str(self.top_class_raw).strip().lower()
        base_loss = 25.0
        if "black rot" in raw: base_loss = 45.0
        elif "brown blight" in raw: base_loss = 30.0
        elif "leaf rust" in raw: base_loss = 15.0
        elif "red spider" in raw: base_loss = 65.0
        elif "mosquito bug" in raw: base_loss = 55.0
        elif "white spot" in raw: base_loss = 10.0
        elif "healthy" in raw or "tea leaf" in raw: return 0.0
        elif not self.is_diseased: return 0.0

        dynamic_loss = base_loss * self.confidence
        minimum_loss_floor = base_loss * 0.4
        return round(max(dynamic_loss, minimum_loss_floor), 1)

    @property
    def severity(self) -> int:
        loss = self.crop_loss_pct
        if loss <= 0.0: return 0
        if loss < 10.0: return 1
        if loss < 25.0: return 2
        if loss < 45.0: return 3
        return 4

    @property
    def all_scores(self) -> dict[str, float]:
        scores = {}
        for d in self.detections:
            if d.confidence > scores.get(d.display_name, 0.0):
                scores[d.display_name] = d.confidence
        return scores

    def annotated_image(self) -> np.ndarray:
        img = (self.preprocessed_img if self.preprocessed_img is not None else self.original_img)
        if img is None: return np.zeros((100, 100, 3), dtype=np.uint8)
        out = img.copy()
        for det in self.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            is_healthy = "tea leaf" in str(det.class_name).lower() or "healthy" in str(det.display_name).lower()
            colour = (0, 200, 80) if is_healthy else (0, 80, 220)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"{det.display_name} {det.confidence:.0%}"
            cv2.putText(out, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        return out

@dataclass
class BatchResult:
    """Aggregated results across all images."""
    results: list[SingleResult] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return len(self.results)

    @property
    def disease_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.top_disease] = counts.get(r.top_disease, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    @property
    def disease_percentages(self) -> dict[str, float]:
        if not self.results: return {}
        n = self.total_images
        return {k: round(v / n * 100, 1) for k, v in self.disease_counts.items()}

    @property
    def diseased_pct(self) -> float:
        if not self.results: return 0.0
        diseased_count = sum(1 for r in self.results if r.is_diseased)
        return round(diseased_count / self.total_images * 100, 1)

    @property
    def estimated_batch_loss_pct(self) -> float:
        if not self.results: return 0.0
        total_loss = sum(r.crop_loss_pct for r in self.results)
        return round(total_loss / self.total_images, 2)

    @property
    def dominant_disease(self) -> str:
        if not self.results: return "None"
        disease_only_counts = {k: v for k, v in self.disease_counts.items()
                               if k not in ["Healthy", "Tea leaf", "No Detection"]}
        if not disease_only_counts: return "Healthy"
        return max(disease_only_counts, key=disease_only_counts.get)

    def summary_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "disease_counts": self.disease_counts,
            "disease_percentages": self.disease_percentages,
            "dominant_disease": self.dominant_disease,
            "avg_confidence": round(sum(r.confidence for r in self.results) / len(self.results), 2) if self.results else 0.0,
            "diseased_pct": self.diseased_pct,
            "batch_loss_pct": self.estimated_batch_loss_pct,
        }

# --- Detector ---

class TeaDiseaseDetector:
    def __init__(self, weights_path: str | Path):
        self.weights_path = Path(weights_path)
        self._model: Optional[YOLO] = None
        self.config_classes: list[str] = CFG["diseases"]["classes"]

    def load(self):
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        logger.info(f"Loading new YOLOv8 model: {self.weights_path}")
        self._model = YOLO(str(self.weights_path))
        logger.success("Model loaded successfully.")

    def _ensure_loaded(self):
        if self._model is None:
            self.load()

    def predict_single(
            self,
            source: str | Path | np.ndarray,
            *,
            preprocess: bool = True,
    ) -> SingleResult:
        self._ensure_loaded()

        if isinstance(source, (str, Path)):
            original = cv2.imread(str(source))
            img_path = str(source)
        else:
            original = source.copy()
            img_path = "live_feed"

        if original is None:
            raise ValueError(f"Could not load image: {source}")

        if preprocess:
            try:
                restored = preprocess_image(original)
            except Exception as e:
                logger.warning(f"Preprocessing failed: {e}. Using resized original.")
                restored = cv2.resize(original, (640, 640))
        else:
            restored = cv2.resize(original, (640, 640))

        # Get settings from config
        conf_thresh = CFG["yolo"].get("conf_thresh", 0.25)
        imgsz = CFG["yolo"].get("imgsz", 640)

        # Run detection
        results = self._model(restored, verbose=False, conf=conf_thresh)
        result = results[0]

        detections: list[Detection] = []
        if result.boxes is not None and len(result.boxes) > 0:
            model_names = result.names

            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_name = model_names.get(cls_id, "Unknown")
                disp = _display_name(raw_name)
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=raw_name,
                    display_name=disp,
                    confidence=conf,
                    bbox_xyxy=xyxy,
                ))

        return SingleResult(
            image_path=img_path,
            detections=detections,
            preprocessed_img=restored,
            original_img=original,
        )

    def predict_batch(
            self,
            paths: list[str | Path],
            *,
            preprocess: bool = True,
            progress_callback=None,
    ) -> BatchResult:
        self._ensure_loaded()
        batch = BatchResult()
        total = len(paths)

        for i, path in enumerate(paths):
            try:
                r = self.predict_single(path, preprocess=preprocess)
                batch.results.append(r)
                logger.debug(f"Processed {Path(path).name}: Found {len(r.detections)} objects.")
            except Exception as exc:
                logger.error(f"Error processing {path}: {exc}")

            if progress_callback:
                progress_callback(i + 1, total)

        return batch
