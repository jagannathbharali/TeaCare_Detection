
from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QApplication, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFileDialog, QProgressBar, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QLineEdit, QTextEdit, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QScrollArea,
    QMessageBox, QStatusBar, QFrame, QSizePolicy,
    QListWidget, QListWidgetItem, QAbstractItemView,
)
from PyQt5.QtGui import QPixmap, QColor, QFont, QIcon
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize
)

from ui.styles import C, GLOBAL_STYLE, card_style, severity_colour
from ui.widgets import (
    ImageViewer, StatCard, SeverityBadge,
    ConfidenceBar, MarkdownTextEdit, SectionHeader,
)
from ui.widgets import (
    ImageViewer, StatCard, SeverityBadge,
    ConfidenceBar, MarkdownTextEdit, SectionHeader,
    ImpactChart  # <-- Add this here!
)


# ═════════════════════════════════════════════════════════════════════════════
#  Background worker threads
# ═════════════════════════════════════════════════════════════════════════════

class DetectionWorker(QThread):
    progress    = pyqtSignal(int, int)          # current, total
    result_ready = pyqtSignal(object)           # BatchResult
    error        = pyqtSignal(str)

    def __init__(self, detector, paths, preprocess=True):
        super().__init__()
        self.detector   = detector
        self.paths      = paths
        self.preprocess = preprocess

    def run(self):
        try:
            batch = self.detector.predict_batch(
                self.paths,
                preprocess=self.preprocess,
                progress_callback=lambda i, t: self.progress.emit(i, t),
            )
            self.result_ready.emit(batch)
        except Exception as e:
            self.error.emit(str(e))


class TrainingWorker(QThread):
    epoch_done  = pyqtSignal(int, int, dict)    # epoch, total, metrics
    finished    = pyqtSignal(str)               # weights path
    error       = pyqtSignal(str)

    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root

    def run(self):
        try:
            from core.trainer import train_model
            def cb(epoch, total, metrics):
                self.epoch_done.emit(epoch, total, metrics)
            weights = train_model(
                self.data_root,
                progress_callback=cb,
            )
            self.finished.emit(str(weights))
        except Exception as e:
            self.error.emit(str(e))


class EconWorker(QThread):
    chunk_ready = pyqtSignal(str)
    finished    = pyqtSignal(str)
    error       = pyqtSignal(str)

    def __init__(self, estimate):
        super().__init__()
        self.estimate = estimate

    def run(self):
        try:
            from core.economics import generate_economic_report
            report = generate_economic_report(
                self.estimate,
                stream_callback=lambda chunk: self.chunk_ready.emit(chunk),
            )
            self.finished.emit(report)
        except Exception as e:
            self.error.emit(str(e))


# ═════════════════════════════════════════════════════════════════════════════
#  Detect Tab
# ═════════════════════════════════════════════════════════════════════════════

class DetectTab(QWidget):
    detection_complete = pyqtSignal(object)   # BatchResult

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._worker: Optional[DetectionWorker] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        root.addWidget(SectionHeader(
            "Disease Detection",
            "Upload images → preprocess → detect → view results"
        ))

        # ── Top controls ──────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.btn_upload = QPushButton("  Upload Images")
        self.btn_upload.setFixedHeight(36)
        self.btn_upload.clicked.connect(self._upload_images)

        self.btn_upload_folder = QPushButton("  Upload Folder")
        self.btn_upload_folder.setProperty("secondary", "true")
        self.btn_upload_folder.setFixedHeight(36)
        self.btn_upload_folder.clicked.connect(self._upload_folder)

        self.chk_preprocess = QCheckBox("Apply Preprocessing")
        self.chk_preprocess.setChecked(True)
        self.chk_preprocess.setStyleSheet(f"color:{C['text_secondary']};")

        self.btn_detect = QPushButton("  Run Detection")
        self.btn_detect.setFixedHeight(36)
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._run_detection)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setProperty("secondary", "true")
        self.btn_clear.setFixedHeight(36)
        self.btn_clear.clicked.connect(self._clear)

        ctrl_row.addWidget(self.btn_upload)
        ctrl_row.addWidget(self.btn_upload_folder)
        ctrl_row.addWidget(self.chk_preprocess)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self.btn_detect)
        ctrl_row.addWidget(self.btn_clear)
        root.addLayout(ctrl_row)

        # ── Progress bar ──────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        # ── Main content splitter ─────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # Left: image list + previews
        left_panel = QWidget()
        left_panel.setStyleSheet(card_style(8))
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        lbl = QLabel("UPLOADED IMAGES")
        lbl.setStyleSheet(
            f"color:{C['text_muted']};font-size:10px;font-weight:700;"
            f"letter-spacing:0.8px;"
        )
        left_layout.addWidget(lbl)

        self.image_list = QListWidget()
        self.image_list.setStyleSheet(f"""
            QListWidget {{
                background: {C['bg']};
                border: 1px solid {C['border']};
                border-radius: 5px;
                color: {C['text_primary']};
                font-size: 11px;
            }}
            QListWidget::item:selected {{
                background: {C['green_deep']};
            }}
            QListWidget::item:hover {{
                background: {C['surface2']};
            }}
        """)
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self.image_list)

        # Before / After previews
        prev_layout = QHBoxLayout()
        self.viewer_orig     = ImageViewer("Original")
        self.viewer_restored = ImageViewer("Preprocessed")
        prev_layout.addWidget(self._labeled_viewer("Original", self.viewer_orig))
        prev_layout.addWidget(self._labeled_viewer("Preprocessed", self.viewer_restored))
        left_layout.addLayout(prev_layout)

        splitter.addWidget(left_panel)

        # Right: results panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Stat cards
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self.card_total    = StatCard("Total Images",    "0")
        self.card_diseased = StatCard("Diseased",        "0%",    colour=C["amber"])
        self.card_dominant = StatCard("Dominant Disease","—")
        self.card_conf     = StatCard("Avg Confidence",  "—",     colour=C["green_light"])
        self.card_severity = StatCard("Severity",        "—")
        for c in [self.card_total, self.card_diseased, self.card_dominant,
                  self.card_conf, self.card_severity]:
            stats_row.addWidget(c)
        right_layout.addLayout(stats_row)

        # Results table
        results_label = QLabel("DETECTION RESULTS")
        results_label.setStyleSheet(
            f"color:{C['text_muted']};font-size:10px;font-weight:700;"
            f"letter-spacing:0.8px;"
        )
        right_layout.addWidget(results_label)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ["Image", "Disease", "Confidence", "Crop Loss Est.", "Severity"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.results_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.results_table.currentCellChanged.connect(
            self._on_table_row_selected
        )
        right_layout.addWidget(self.results_table)

        # Confidence breakdown (per-class bars)
        breakdown_box = QGroupBox("Confidence Breakdown — Selected Image")
        bb_layout = QVBoxLayout(breakdown_box)
        bb_layout.setSpacing(4)
        self._conf_bars: list[ConfidenceBar] = []
        self._build_conf_bars(bb_layout)
        right_layout.addWidget(breakdown_box)

        splitter.addWidget(right_panel)
        splitter.setSizes([350, 700])
        root.addWidget(splitter)

    # ── helpers ──────────────────────────────────────────────────────────
    def _labeled_viewer(self, label: str, viewer: ImageViewer) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{C['bg']};border-radius:5px;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color:{C['text_muted']};font-size:9px;font-weight:600;"
            f"letter-spacing:0.5px;text-align:center;"
        )
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        lay.addWidget(viewer)
        return w

    def _build_conf_bars(self, layout: QVBoxLayout):
        import yaml as _yaml
        cfg = _yaml.safe_load(
            open(Path(__file__).parent.parent / "config.yaml")
        )
        
        # Keep track of added names to prevent duplicates
        added_names = set()
        
        for raw_cls in cfg["diseases"]["classes"]:
            # Grab the clean display name from config, fallback to raw if not found
            clean_name = cfg["diseases"].get("display_names", {}).get(raw_cls, raw_cls)
            
            # Create the bar using the clean name
            if clean_name not in added_names:
                bar = ConfidenceBar(clean_name, 0.0)
                layout.addWidget(bar)
                self._conf_bars.append(bar)
                added_names.add(clean_name)

    # ── slots ─────────────────────────────────────────────────────────────
    def _upload_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if files:
            self._load_files(files)

    def _upload_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            files = [
                str(p) for p in Path(folder).rglob("*")
                if p.suffix.lower() in exts
            ]
            if files:
                self._load_files(files)
            else:
                QMessageBox.warning(self, "No Images",
                    "No image files found in selected folder.")

    def _load_files(self, files: list[str]):
        self.app_state["image_paths"] = files
        self.image_list.clear()
        for f in files:
            item = QListWidgetItem(Path(f).name)
            item.setData(Qt.UserRole, f)
            self.image_list.addItem(item)
        self.btn_detect.setEnabled(
            bool(self.app_state.get("model_path"))
        )
        if not self.app_state.get("model_path"):
            self._status(f"Loaded {len(files)} images. ⚠ No model loaded.")
        else:
            self._status(f"Loaded {len(files)} images. Ready to detect.")

    def _on_image_selected(self, row: int):
        if row < 0:
            return
        paths = self.app_state.get("image_paths", [])
        if row >= len(paths):
            return
        path = paths[row]
        self.viewer_orig.set_path(path)

        # Show preprocessed if available
        results = self.app_state.get("batch_result")
        if results and row < len(results.results):
            r = results.results[row]
            if r.preprocessed_img is not None:
                self.viewer_restored.set_image(r.preprocessed_img)
                
            # Update confidence bars cleanly
            for bar in self._conf_bars:
                score = r.all_scores.get(bar._label, 0.0)
                bar.set_confidence(bar._label, score)

    def _on_table_row_selected(self, row, *args):
        self._on_image_selected(row)

    def _run_detection(self):
        paths = self.app_state.get("image_paths", [])
        detector = self.app_state.get("detector")
        if not paths:
            QMessageBox.warning(self, "No Images", "Please upload images first.")
            return
        if not detector:
            QMessageBox.warning(self, "No Model",
                "No model loaded. Train a model or load weights in Settings.")
            return

        self.btn_detect.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(paths))
        self.progress_bar.setValue(0)
        self._status("Running detection ...")

        self._worker = DetectionWorker(
            detector, paths,
            preprocess=self.chk_preprocess.isChecked()
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_detection_done)
        self._worker.error.connect(self._on_detection_error)
        self._worker.start()

    def _on_progress(self, current: int, total: int):
        self.progress_bar.setValue(current)
        self._status(f"Processing image {current}/{total} ...")

    def _on_detection_done(self, batch_result):
        self.app_state["batch_result"] = batch_result
        self.progress_bar.setVisible(False)
        self.btn_detect.setEnabled(True)

        summary = batch_result.summary_dict()

        # Update stat cards
        self.card_total.update_value(str(summary["total_images"]))
        self.card_diseased.update_value(
            f"{summary['diseased_pct']:.1f}%",
            colour=C["amber"] if summary["diseased_pct"] > 20 else C["green_light"]
        )
        self.card_dominant.update_value(summary["dominant_disease"])
        self.card_conf.update_value(f"{summary['avg_confidence']*100:.1f}%")

        # Compute severity from loss
        loss = summary["batch_loss_pct"]
        if loss < 10:
            sev = "Low"
        elif loss < 25:
            sev = "Moderate"
        elif loss < 45:
            sev = "High"
        else:
            sev = "Critical"
        self.card_severity.update_value(sev, colour=severity_colour(sev))

        # Populate table
        self.results_table.setRowCount(0)
        for r in batch_result.results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            name_item = QTableWidgetItem(Path(r.image_path).name)
            dis_item  = QTableWidgetItem(r.top_disease)
            conf_item = QTableWidgetItem(f"{r.confidence*100:.1f}%")
            loss_item = QTableWidgetItem(f"{r.crop_loss_pct:.0f}%")
            sev_item  = QTableWidgetItem(
                ["—","Low","Moderate","High","Critical"][min(r.severity,4)]
            )

            # Colour diseased rows
            if r.is_diseased:
                for item in [name_item, dis_item, conf_item, loss_item, sev_item]:
                    item.setForeground(QColor(C["amber"]))

            self.results_table.setItem(row, 0, name_item)
            self.results_table.setItem(row, 1, dis_item)
            self.results_table.setItem(row, 2, conf_item)
            self.results_table.setItem(row, 3, loss_item)
            self.results_table.setItem(row, 4, sev_item)

        self._status(
            f"Detection complete. {summary['diseased_pct']:.1f}% diseased. "
            f"Dominant: {summary['dominant_disease']}"
        )
        self.detection_complete.emit(batch_result)

    def _on_detection_error(self, error: str):
        self.progress_bar.setVisible(False)
        self.btn_detect.setEnabled(True)
        QMessageBox.critical(self, "Detection Error", error)
        self._status(f"Error: {error}")

    def _clear(self):
        self.app_state.pop("image_paths", None)
        self.app_state.pop("batch_result", None)
        self.image_list.clear()
        self.results_table.setRowCount(0)
        self.viewer_orig.set_image(None)
        self.viewer_restored.set_image(None)
        for bar in self._conf_bars:
            bar.set_confidence(bar._label, 0.0)
        for c in [self.card_total, self.card_diseased, self.card_dominant,
                  self.card_conf, self.card_severity]:
            c.update_value("—")
        self._status("Cleared.")

    def _status(self, msg: str):
        # propagate to main window statusbar
        window = self.window()
        if hasattr(window, "statusBar"):
            window.statusBar().showMessage(msg, 8000)


# ═════════════════════════════════════════════════════════════════════════════
#  Economics Tab
# ═════════════════════════════════════════════════════════════════════════════

class EconomicsTab(QWidget):
    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._worker: Optional[EconWorker] = None
        self._full_report: str = ""
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        root.addWidget(SectionHeader(
            "Economic Impact Analysis",
            "AI-generated report powered by local Mistral LLM"
        ))

        # ── Financial overview cards ───────────────────────────────────────
        fin_box = QGroupBox("Financial Snapshot")
        fin_lay = QHBoxLayout(fin_box)
        fin_lay.setSpacing(8)

        # These are the lines that accidentally got deleted!
        self.ecard_gross  = StatCard("Gross Crop Value",   "—", "USD", colour=C["green_light"])
        self.ecard_loss   = StatCard("Estimated Loss",      "—", "USD", colour=C["red"])
        self.ecard_net    = StatCard("Net Value",           "—", "USD", colour=C["amber"])
        self.ecard_losspct= StatCard("Loss %",              "—", "%",   colour=C["amber"])
        self.ecard_sev    = StatCard("Severity",            "—")

        for c in [self.ecard_gross, self.ecard_loss, self.ecard_net,
                  self.ecard_losspct, self.ecard_sev]:
            fin_lay.addWidget(c)
        root.addWidget(fin_box)

        # The new Impact Chart we added
        self.impact_chart = ImpactChart()
        root.addWidget(self.impact_chart)

        # ── Generate button ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("  Generate Economic Report (Mistral)")
        self.btn_generate.setFixedHeight(38)
        self.btn_generate.clicked.connect(self._generate_report)

        self.btn_refresh = QPushButton("Refresh from Detection")
        self.btn_refresh.setProperty("secondary", "true")
        self.btn_refresh.setFixedHeight(38)
        self.btn_refresh.clicked.connect(self._refresh_estimates)

        btn_row.addWidget(self.btn_refresh)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_generate)
        root.addLayout(btn_row)

        self.gen_progress = QProgressBar()
        self.gen_progress.setRange(0, 0)   # indeterminate
        self.gen_progress.setFixedHeight(8)
        self.gen_progress.setVisible(False)
        root.addWidget(self.gen_progress)

        # ── Report display ─────────────────────────────────────────────────
        report_label = QLabel("AI ECONOMIC REPORT")
        report_label.setStyleSheet(
            f"color:{C['text_muted']};font-size:10px;font-weight:700;"
            f"letter-spacing:0.8px;"
        )
        root.addWidget(report_label)

        self.report_view = MarkdownTextEdit()
        self.report_view.setPlaceholderText(
            "Run detection first, then click 'Generate Economic Report' "
            "to get a detailed AI analysis powered by local Mistral LLM.\n\n"
            "Make sure Ollama is running:\n  ollama serve\n  ollama pull mistral"
        )
        root.addWidget(self.report_view)

    def refresh_for_batch(self, batch_result):
        """Called when detection completes."""
        self._refresh_estimates()

    def _refresh_estimates(self):
        batch = self.app_state.get("batch_result")
        if not batch:
            return
        from core.economics import compute_economic_estimate
        summary  = batch.summary_dict()
        estimate = compute_economic_estimate(summary)
        self.app_state["economic_estimate"] = estimate

        sym = "$"
        self.ecard_gross.update_value(f"{sym}{estimate.gross_crop_value_usd:,.0f}")
        self.ecard_loss.update_value(
            f"{sym}{estimate.estimated_loss_usd:,.0f}",
            colour=C["red"]
        )
        self.ecard_net.update_value(f"{sym}{estimate.net_value_usd:,.0f}")
        self.ecard_losspct.update_value(
            f"{estimate.batch_loss_pct:.1f}",
            colour=severity_colour(estimate.severity_label)
        )
        self.ecard_sev.update_value(
            estimate.severity_label,
            colour=severity_colour(estimate.severity_label)
        )
        self.ecard_losspct.update_value(
            f"{estimate.batch_loss_pct:.1f}",
            colour=severity_colour(estimate.severity_label)
        )
        self.ecard_sev.update_value(
            estimate.severity_label,
            colour=severity_colour(estimate.severity_label)
        )

        # 👇 ADD THIS BLOCK RIGHT HERE 👇
        self.impact_chart.update_chart(
            gross = estimate.gross_crop_value_usd,
            net = estimate.net_value_usd,
            loss = estimate.estimated_loss_usd,
            loss_pct = estimate.batch_loss_pct
        )
        # 👆 ───────────────────────── 👆

    def _generate_report(self):
        estimate = self.app_state.get("economic_estimate")
        if not estimate:
            self._refresh_estimates()
            estimate = self.app_state.get("economic_estimate")
        if not estimate:
            QMessageBox.warning(self, "No Data",
                "Run detection first to generate economic estimates.")
            return

        self.btn_generate.setEnabled(False)
        self.gen_progress.setVisible(True)
        self.report_view.clear()
        self.report_view.setPlainText("Generating report with Mistral...")

        self._worker = EconWorker(estimate)
        self._worker.chunk_ready.connect(self._on_chunk)
        self._worker.finished.connect(self._on_report_done)
        self._worker.error.connect(self._on_report_error)
        self._worker.start()

    def _on_chunk(self, chunk: str):
        self.report_view.append_chunk(chunk)

    def _on_report_done(self, full_report: str):
        self._full_report = full_report
        self.app_state["economic_report"] = full_report
        self.gen_progress.setVisible(False)
        self.btn_generate.setEnabled(True)
        self.report_view.set_markdown(full_report)

    def _on_report_error(self, error: str):
        self.gen_progress.setVisible(False)
        self.btn_generate.setEnabled(True)
        self.report_view.setPlainText(f"Error: {error}")


# ═════════════════════════════════════════════════════════════════════════════
#  Train Tab
# ═════════════════════════════════════════════════════════════════════════════

class TrainTab(QWidget):
    model_ready = pyqtSignal(str)   # weights path

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._worker: Optional[TrainingWorker] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        root.addWidget(SectionHeader(
            "Model Training",
            "Train YOLOv8 classifier on your tea disease dataset"
        ))

        # Dataset path
        ds_box = QGroupBox("Dataset")
        ds_lay = QHBoxLayout(ds_box)
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("Path to Tea Leaves Dataset/ folder (contains data.yaml)")
        self.data_path_edit.setText(str(Path("Tea Leaves Dataset").absolute()))
        btn_browse_ds = QPushButton("Browse")
        btn_browse_ds.setFixedWidth(80)
        btn_browse_ds.clicked.connect(self._browse_data)
        btn_scan = QPushButton("Scan Dataset")
        btn_scan.setFixedWidth(110)
        btn_scan.clicked.connect(self._scan_dataset)
        ds_lay.addWidget(self.data_path_edit)
        ds_lay.addWidget(btn_browse_ds)
        ds_lay.addWidget(btn_scan)
        root.addWidget(ds_box)

        # Dataset info card
        self.dataset_info = QLabel("Click 'Scan Dataset' to verify folder structure.")
        self.dataset_info.setStyleSheet(
            f"background:{C['surface']}; border:1px solid {C['border']};"
            f"border-radius:6px; padding:8px 12px; color:{C['text_secondary']};"
            f"font-size:11px; font-family: 'Courier New', monospace;"
        )
        self.dataset_info.setWordWrap(True)
        root.addWidget(self.dataset_info)

        # Existing weights
        wt_box = QGroupBox("Load Existing Weights (skip training)")
        wt_lay = QHBoxLayout(wt_box)
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("Path to best.pt weights file")
        btn_browse_wt = QPushButton("Browse")
        btn_browse_wt.setFixedWidth(80)
        btn_browse_wt.clicked.connect(self._browse_weights)
        btn_load = QPushButton("Load Model")
        btn_load.setFixedWidth(100)
        btn_load.clicked.connect(self._load_model)
        wt_lay.addWidget(self.weights_edit)
        wt_lay.addWidget(btn_browse_wt)
        wt_lay.addWidget(btn_load)
        root.addWidget(wt_box)

        # Train button + progress
        train_row = QHBoxLayout()
        self.btn_train = QPushButton("  Start Training")
        self.btn_train.setFixedHeight(40)
        self.btn_train.clicked.connect(self._start_training)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setProperty("secondary", "true")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setEnabled(False)
        train_row.addWidget(self.btn_train)
        train_row.addWidget(self.btn_stop)
        root.addLayout(train_row)

        self.epoch_bar = QProgressBar()
        self.epoch_bar.setFixedHeight(14)
        self.epoch_bar.setVisible(False)
        root.addWidget(self.epoch_bar)

        # Training log
        log_label = QLabel("TRAINING LOG")
        log_label.setStyleSheet(
            f"color:{C['text_muted']};font-size:10px;font-weight:700;"
            f"letter-spacing:0.8px;"
        )
        root.addWidget(log_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(f"""
            QTextEdit {{
                background: {C['surface']};
                border: 1px solid {C['border']};
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                color: {C['green_light']};
                padding: 8px;
            }}
        """)
        root.addWidget(self.log_view)

    def _browse_data(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Tea Leaves Dataset Folder")
        if folder:
            self.data_path_edit.setText(folder)
            self._scan_dataset()

    def _scan_dataset(self):
        """
        Discover and display Roboflow dataset structure.
        Reads data.yaml directly — no folder renaming needed.
        """
        path = self.data_path_edit.text().strip()
        if not path:
            return
        try:
            from core.trainer import find_data_yaml, load_roboflow_yaml, validate_dataset
            from pathlib import Path as _P
            root      = _P(path)
            yaml_path = find_data_yaml(root)
            data      = load_roboflow_yaml(yaml_path)

            # Count images per split
            _IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}
            def _count(key):
                raw = data.get(key, "")
                if not raw: return 0
                d = _P(raw)
                if not d.exists(): d = root / raw
                if not d.exists():
                    cands = list(root.rglob(f"{key}/images"))
                    d = cands[0] if cands else d
                return len([f for f in d.glob("*") if f.suffix.lower() in _IMG_EXTS]) \
                       if d.exists() else 0

            n_train = _count("train")
            n_val   = _count("val")
            n_test  = _count("test")

            cfg_d    = self.app_state.get("cfg_diseases", {})
            disp_map = cfg_d.get("display_names", {})
            class_str = "  ".join(
                f"{n}({disp_map.get(n,n)})" for n in data["names"]
            )

            lines = [
                f"✓  data.yaml : {yaml_path.name}",
                f"   Root      : {root.resolve()}",
                f"   Train     : {n_train} images",
                f"   Val       : {n_val} images",
                f"   Test      : {n_test} images",
                f"   nc={data['nc']}  classes: {class_str}",
                f"   Format    : Roboflow detection (images + labels)",
            ]
            self.dataset_info.setText("\n".join(lines))
            self.dataset_info.setStyleSheet(
                f"background:{C['surface']}; border:1px solid {C['green_mid']};"
                f"border-radius:6px; padding:8px 12px; color:{C['green_light']};"
                f"font-size:11px; font-family: 'Courier New', monospace;"
            )
            total = n_train + n_val + n_test
            self.log_view.append(
                f"✓ Dataset ready: nc={data['nc']} classes, {total} total images."
            )
        except Exception as e:
            self.dataset_info.setText(f"✗  {e}")
            self.dataset_info.setStyleSheet(
                f"background:{C['surface']}; border:1px solid {C['red']};"
                f"border-radius:6px; padding:8px 12px; color:{C['red']};"
                f"font-size:11px; font-family: 'Courier New', monospace;"
            )
            self.log_view.append(f"✗ Scan failed: {e}")

    def _browse_weights(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Weights", "", "PyTorch weights (*.pt)"
        )
        if f:
            self.weights_edit.setText(f)

    def _load_model(self):
        path = self.weights_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Not Found", "Weights file not found.")
            return
        self._init_detector(path)

    def _init_detector(self, weights_path: str):
        from core.detector import TeaDiseaseDetector
        try:
            det = TeaDiseaseDetector(weights_path)
            det.load()
            self.app_state["detector"]    = det
            self.app_state["model_path"]  = weights_path
            self.log_view.append(f"✓ Model loaded: {weights_path}")
            self.model_ready.emit(weights_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _start_training(self):
        data_root = self.data_path_edit.text().strip()
        if not data_root or not Path(data_root).exists():
            QMessageBox.warning(self, "No Data", "Select a valid data folder.")
            return

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.epoch_bar.setVisible(True)

        import yaml
        total_epochs = yaml.safe_load(
            open(Path(__file__).parent.parent / "config.yaml")
        )["yolo"]["epochs"]
        self.epoch_bar.setRange(0, total_epochs)
        self.epoch_bar.setValue(0)

        self.log_view.clear()
        self.log_view.append("Starting training ...")

        self._worker = TrainingWorker(data_root)
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.finished.connect(self._on_train_done)
        self._worker.error.connect(self._on_train_error)
        self._worker.start()

    def _on_epoch(self, epoch: int, total: int, metrics: dict):
        self.epoch_bar.setValue(epoch)
        msg_parts = [f"Epoch {epoch}/{total}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                msg_parts.append(f"{k}={v:.4f}")
        self.log_view.append("  ".join(msg_parts))

    def _on_train_done(self, weights_path: str):
        self.epoch_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.weights_edit.setText(weights_path)
        self.log_view.append(f"\n✓ Training complete!\n  Best weights: {weights_path}")
        self._init_detector(weights_path)

    def _on_train_error(self, error: str):
        self.epoch_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log_view.append(f"\n✗ Error: {error}")
        QMessageBox.critical(self, "Training Error", error)


# ═════════════════════════════════════════════════════════════════════════════
#  Export Tab
# ═════════════════════════════════════════════════════════════════════════════

class ExportTab(QWidget):
    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        root.addWidget(SectionHeader(
            "Export Reports",
            "Save detection results and economic analysis as PDF or JSON"
        ))

        # Output directory
        out_box = QGroupBox("Output Directory")
        out_lay = QHBoxLayout(out_box)
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setText(str(Path("reports").absolute()))
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse_output)
        out_lay.addWidget(self.out_dir_edit)
        out_lay.addWidget(btn_browse)
        root.addWidget(out_box)

        # Export buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_export_pdf = QPushButton("  Export PDF Report")
        self.btn_export_pdf.setFixedHeight(44)
        self.btn_export_pdf.clicked.connect(self._export_pdf)

        self.btn_export_json = QPushButton("  Export JSON Data")
        self.btn_export_json.setProperty("secondary", "true")
        self.btn_export_json.setFixedHeight(44)
        self.btn_export_json.clicked.connect(self._export_json)

        btn_row.addWidget(self.btn_export_pdf)
        btn_row.addWidget(self.btn_export_json)
        root.addLayout(btn_row)

        # Export log
        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setMaximumHeight(200)
        self.export_log.setStyleSheet(f"""
            QTextEdit {{
                background: {C['surface']};
                border: 1px solid {C['border']};
                border-radius: 5px;
                font-size: 11px;
                color: {C['text_primary']};
                padding: 8px;
            }}
        """)
        root.addWidget(self.export_log)
        root.addStretch()

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.out_dir_edit.setText(folder)

    def _get_results_list(self) -> list[dict]:
        batch = self.app_state.get("batch_result")
        if not batch:
            return []
        return [
            {
                "image_path":  r.image_path,
                "top_disease": r.top_disease,
                "confidence":  r.confidence,
                "all_scores":  r.all_scores,
                "crop_loss_pct": r.crop_loss_pct,
                "severity":    r.severity,
            }
            for r in batch.results
        ]

    def _export_pdf(self):
        batch    = self.app_state.get("batch_result")
        estimate = self.app_state.get("economic_estimate")
        if not batch:
            QMessageBox.warning(self, "No Data", "Run detection first.")
            return

        if not estimate:
            from core.economics import compute_economic_estimate
            estimate = compute_economic_estimate(batch.summary_dict())
            self.app_state["economic_estimate"] = estimate

        report_md = self.app_state.get(
            "economic_report",
            "*(Economic report not yet generated — run from the Economics tab)*"
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.out_dir_edit.text())
        out_path = out_dir / f"TeaGuard_Report_{ts}.pdf"

        try:
            from utils.report_exporter import export_pdf
            export_pdf(
                out_path,
                batch.summary_dict(),
                estimate,
                report_md,
                self._get_results_list(),
            )
            self.export_log.append(f"✓ PDF saved: {out_path}")
            QMessageBox.information(self, "Exported", f"PDF saved:\n{out_path}")
        except Exception as e:
            self.export_log.append(f"✗ PDF error: {e}")
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_json(self):
        batch    = self.app_state.get("batch_result")
        estimate = self.app_state.get("economic_estimate")
        if not batch:
            QMessageBox.warning(self, "No Data", "Run detection first.")
            return
        if not estimate:
            from core.economics import compute_economic_estimate
            estimate = compute_economic_estimate(batch.summary_dict())

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.out_dir_edit.text())
        out_path = out_dir / f"TeaGuard_Data_{ts}.json"

        try:
            from utils.report_exporter import export_json
            export_json(out_path, batch.summary_dict(), estimate,
                        self._get_results_list())
            self.export_log.append(f"✓ JSON saved: {out_path}")
            QMessageBox.information(self, "Exported", f"JSON saved:\n{out_path}")
        except Exception as e:
            self.export_log.append(f"✗ JSON error: {e}")
            QMessageBox.critical(self, "Export Error", str(e))


# ═════════════════════════════════════════════════════════════════════════════
#  Settings Tab
# ═════════════════════════════════════════════════════════════════════════════

class SettingsTab(QWidget):
    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._cfg_path = Path(__file__).parent.parent / "config.yaml"
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        root.addWidget(SectionHeader("Settings", "Configure application parameters"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"border: none; background: transparent;")
        content = QWidget()
        scroll.setWidget(content)
        lay = QVBoxLayout(content)
        lay.setSpacing(12)

        # ── LLM settings ──────────────────────────────────────────────────
        llm_box = QGroupBox("LLM / Ollama")
        llm_lay = QGridLayout(llm_box)
        llm_lay.addWidget(QLabel("Ollama Host:"), 0, 0)
        self.ollama_host = QLineEdit("http://localhost:11434")
        llm_lay.addWidget(self.ollama_host, 0, 1)
        llm_lay.addWidget(QLabel("Model:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["mistral", "llama3", "phi3", "gemma"])
        llm_lay.addWidget(self.model_combo, 1, 1)
        llm_lay.addWidget(QLabel("Timeout (s):"), 2, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 600)
        self.timeout_spin.setValue(120)
        llm_lay.addWidget(self.timeout_spin, 2, 1)
        lay.addWidget(llm_box)

        # ── Economics settings ─────────────────────────────────────────────
        eco_box = QGroupBox("Economics Baseline")
        eco_lay = QGridLayout(eco_box)
        eco_lay.addWidget(QLabel("Yield (kg/hectare):"), 0, 0)
        self.yield_spin = QDoubleSpinBox()
        self.yield_spin.setRange(500, 10000)
        self.yield_spin.setValue(2500)
        eco_lay.addWidget(self.yield_spin, 0, 1)
        eco_lay.addWidget(QLabel("Price per kg (USD):"), 1, 0)
        self.price_spin = QDoubleSpinBox()
        self.price_spin.setRange(0.5, 50)
        self.price_spin.setValue(3.50)
        self.price_spin.setSingleStep(0.1)
        eco_lay.addWidget(self.price_spin, 1, 1)
        eco_lay.addWidget(QLabel("Estate size (hectares):"), 2, 0)
        self.hectares_spin = QDoubleSpinBox()
        self.hectares_spin.setRange(1, 5000)
        self.hectares_spin.setValue(50)
        eco_lay.addWidget(self.hectares_spin, 2, 1)
        lay.addWidget(eco_box)

        # Save button
        btn_save = QPushButton("Save Settings")
        btn_save.setFixedHeight(36)
        btn_save.clicked.connect(self._save_settings)
        lay.addWidget(btn_save)
        lay.addStretch()

        root.addWidget(scroll)

    def _save_settings(self):
        import yaml
        with open(self._cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["llm"]["host"]    = self.ollama_host.text()
        cfg["llm"]["model"]   = self.model_combo.currentText()
        cfg["llm"]["timeout"] = self.timeout_spin.value()
        cfg["economics"]["avg_yield_kg_per_hectare"] = self.yield_spin.value()
        cfg["economics"]["avg_price_per_kg_usd"]     = self.price_spin.value()
        cfg["economics"]["avg_estate_hectares"]      = self.hectares_spin.value()
        with open(self._cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        QMessageBox.information(self, "Saved", "Settings saved to config.yaml")


# ═════════════════════════════════════════════════════════════════════════════
#  Main Window
# ═════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍃 TeaGuard AI — Tea Disease Detection")
        self.resize(1280, 820)
        self.setMinimumSize(1000, 700)

        # Shared application state
        self._state: dict = {
            "image_paths":       [],
            "batch_result":      None,
            "detector":          None,
            "model_path":        None,
            "economic_estimate": None,
            "economic_report":   None,
        }

        self._build_ui()
        self._try_autoload_model()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {C['surface']}, stop:1 {C['surface2']});"
            f"border-bottom: 1px solid {C['border']};"
        )
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(20, 0, 20, 0)

        logo = QLabel("🍃 TeaGuard AI")
        logo.setStyleSheet(
            f"color: {C['green_light']}; font-size: 17px; font-weight: 700; "
            f"letter-spacing: 0.5px;"
        )
        h_lay.addWidget(logo)

        subtitle = QLabel("Tea Leaf Disease Detection & Economic Impact Predictor")
        subtitle.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 11px; margin-left: 12px;"
        )
        h_lay.addWidget(subtitle)
        h_lay.addStretch()

        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet(
            f"color: {C['amber']}; font-size: 11px; font-weight: 600;"
        )
        h_lay.addWidget(self.model_status)

        root.addWidget(header)

        # ── Tabs ──────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.detect_tab   = DetectTab(self._state)
        self.train_tab    = TrainTab(self._state)
        self.econ_tab     = EconomicsTab(self._state)
        self.export_tab   = ExportTab(self._state)
        self.settings_tab = SettingsTab(self._state)

        self.tabs.addTab(self.detect_tab,   "  Detect  ")
        self.tabs.addTab(self.train_tab,    "  Train  ")
        self.tabs.addTab(self.econ_tab,     "  Economics  ")
        self.tabs.addTab(self.export_tab,   "  Export  ")
        self.tabs.addTab(self.settings_tab, "  Settings  ")

        root.addWidget(self.tabs)

        # ── Status bar ────────────────────────────────────────────────────
        sb = QStatusBar()
        sb.setStyleSheet(
            f"background: {C['surface']}; color: {C['text_secondary']};"
            f"border-top: 1px solid {C['border']}; font-size: 11px;"
        )
        self.setStatusBar(sb)
        sb.showMessage("Ready. Load a model or start training to begin.")

        # ── Wire signals ──────────────────────────────────────────────────
        self.detect_tab.detection_complete.connect(
            self.econ_tab.refresh_for_batch
        )
        self.detect_tab.detection_complete.connect(
            lambda _: self.tabs.setCurrentIndex(2)
        )
        self.train_tab.model_ready.connect(self._on_model_ready)

    def _try_autoload_model(self):
        """Auto-load best.pt if it exists, and auto-scan dataset."""
        # Auto-scan Tea_Dataset
        tea_dataset = Path("Tea Leaves Dataset")
        if tea_dataset.exists():
            QTimer.singleShot(500, self.train_tab._scan_dataset)

        # Auto-load weights if already trained
        default_path = Path("models/tea_disease_det/weights/best.pt")
        if default_path.exists():
            from core.detector import TeaDiseaseDetector
            try:
                det = TeaDiseaseDetector(default_path)
                det.load()
                self._state["detector"]   = det
                self._state["model_path"] = str(default_path)
                self._on_model_ready(str(default_path))
            except Exception:
                pass

    def _on_model_ready(self, weights_path: str):
        self.model_status.setText(f"✓ Model: {Path(weights_path).name}")
        self.model_status.setStyleSheet(
            f"color: {C['green_light']}; font-size: 11px; font-weight: 600;"
        )
        self.detect_tab.btn_detect.setEnabled(
            bool(self._state.get("image_paths"))
        )
        self.statusBar().showMessage(f"Model loaded: {weights_path}", 5000)


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def launch():
    # 1. Set High-DPI support FIRST
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 2. THEN create the app
    app = QApplication(sys.argv)
    app.setApplicationName("TeaGuard AI")
    app.setStyleSheet(GLOBAL_STYLE)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
