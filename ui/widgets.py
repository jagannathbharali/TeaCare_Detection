from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame,
    QPushButton, QSizePolicy, QGraphicsDropShadowEffect,
    QTextEdit,
)
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPainter, QFont, QPen,
    QLinearGradient, QBrush, QFontDatabase,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QPropertyAnimation, QRect

from ui.styles import C, severity_colour, card_style


# ═════════════════════════════════════════════════════════════════════════════
#  ImageViewer — show opencv BGR array with optional overlay label
# ═════════════════════════════════════════════════════════════════════════════

class ImageViewer(QLabel):
    """
    Displays a BGR numpy array scaled to fit the widget.
    Shows a placeholder when no image is loaded.
    """
    clicked = pyqtSignal()

    def __init__(self, placeholder_text: str = "No image loaded",
                 parent=None):
        super().__init__(parent)
        self._placeholder = placeholder_text
        self._pixmap      = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"""
            background-color: {C['surface']};
            border: 1px dashed {C['border']};
            border-radius: 8px;
            color: {C['text_muted']};
            font-size: 12px;
        """)
        self.setText(placeholder_text)
        self.setCursor(Qt.PointingHandCursor)

    def set_image(self, img_bgr: np.ndarray | None):
        if img_bgr is None:
            self._pixmap = None
            self.setText(self._placeholder)
            return
        rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_display()
        self.setText("")

    def set_path(self, path: str | Path):
        img = cv2.imread(str(path))
        if img is not None:
            self.set_image(img)

    def _update_display(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        self._update_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


# ═════════════════════════════════════════════════════════════════════════════
#  StatCard — metric display tile
# ═════════════════════════════════════════════════════════════════════════════

class StatCard(QWidget):
    def __init__(self, title: str, value: str = "—",
                 unit: str = "", colour: str = C["green_light"],
                 parent=None):
        super().__init__(parent)
        self.setStyleSheet(card_style(10))
        self.setMinimumWidth(130)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet(
            f"color: {C['text_secondary']}; font-size: 10px; "
            f"font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase;"
        )

        self._value_lbl = QLabel(value)
        self._value_lbl.setStyleSheet(
            f"color: {colour}; font-size: 22px; font-weight: 700;"
        )

        self._unit_lbl = QLabel(unit)
        self._unit_lbl.setStyleSheet(
            f"color: {C['text_muted']}; font-size: 10px;"
        )

        layout.addWidget(self._title_lbl)
        layout.addWidget(self._value_lbl)
        if unit:
            layout.addWidget(self._unit_lbl)

    def update_value(self, value: str, colour: str | None = None):
        self._value_lbl.setText(value)
        if colour:
            self._value_lbl.setStyleSheet(
                f"color: {colour}; font-size: 22px; font-weight: 700;"
            )


# ═════════════════════════════════════════════════════════════════════════════
#  SeverityBadge
# ═════════════════════════════════════════════════════════════════════════════

class SeverityBadge(QLabel):
    def __init__(self, label: str = "—", parent=None):
        super().__init__(parent)
        self.set_severity(label)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(26)
        self.setMinimumWidth(80)

    def set_severity(self, label: str):
        self.setText(label)
        colour = severity_colour(label)
        self.setStyleSheet(f"""
            background-color: transparent;
            border: 1.5px solid {colour};
            border-radius: 13px;
            color: {colour};
            padding: 2px 12px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)


# ═════════════════════════════════════════════════════════════════════════════
#  ConfidenceBar — horizontal bar showing confidence %
# ═════════════════════════════════════════════════════════════════════════════

class ConfidenceBar(QWidget):
    def __init__(self, label: str, confidence: float = 0.0, parent=None):
        super().__init__(parent)
        self._label = label
        self._conf  = confidence
        self.setFixedHeight(22)
        self.setMinimumWidth(200)

    def set_confidence(self, label: str, confidence: float):
        self._label = label
        self._conf  = max(0.0, min(1.0, confidence))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.setBrush(QBrush(QColor(C["surface"])))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, 4, 4)

        # Fill
        fill_w = int(w * self._conf)
        if fill_w > 0:
            grad = QLinearGradient(0, 0, fill_w, 0)
            grad.setColorAt(0, QColor(C["green_deep"]))
            grad.setColorAt(1, QColor(C["green_light"]))
            painter.setBrush(QBrush(grad))
            painter.drawRoundedRect(0, 0, fill_w, h, 4, 4)

        # Text
        painter.setPen(QColor(C["text_primary"]))
        painter.setFont(QFont("Segoe UI", 8))
        label_text = f"{self._label[:20]}  {self._conf*100:.1f}%"
        painter.drawText(QRect(6, 0, w - 12, h), Qt.AlignVCenter, label_text)


# ═════════════════════════════════════════════════════════════════════════════
#  MarkdownTextEdit — renders (basic) markdown in a QTextEdit
# ═════════════════════════════════════════════════════════════════════════════

class MarkdownTextEdit(QTextEdit):
    """
    Display markdown-formatted text with resilient formatting
    (headings, bold, bullets, and proper tables).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {C['surface']};
                border: 1px solid {C['border']};
                border-radius: 6px;
                padding: 12px;
                color: {C['text_primary']};
                font-size: 13px;
                line-height: 1.6;
            }}
        """)

    def set_markdown(self, text: str):
        lines = text.splitlines()
        html  = []
        in_table = False

        for line in lines:
            s = line.strip()

            # Handle Table Open/Close/Rows
            if s.startswith("|"):
                if not in_table:
                    html.append(f"<table style='border-collapse:collapse; margin:10px 0; width:100%;'>")
                    in_table = True
                if "---" in s:
                    continue  # Skip markdown separator line
                cells = [c.strip() for c in s.split("|")[1:-1]]
                if cells:
                    row = "".join(f"<td style='padding:6px 10px; border:1px solid {C['border']};'>{c}</td>" for c in cells)
                    html.append(f"<tr>{row}</tr>")
                continue
            else:
                if in_table:
                    html.append("</table>")
                    in_table = False

            if not s:
                html.append("<br>")
                continue

            # Catch strict headings (##) AND numbered headings (1. Text)
            if s.startswith("## "):
                html.append(f'<h3 style="color:{C["green_light"]};margin:12px 0 4px;">{s[3:]}</h3>')
            elif s.startswith("# "):
                html.append(f'<h2 style="color:{C["green_light"]};margin:14px 0 6px;">{s[2:]}</h2>')
            elif s[0].isdigit() and len(s) > 2 and s[1:3] == ". ":
                html.append(f'<h3 style="color:{C["green_light"]};margin:12px 0 4px;">{s}</h3>')

            # Catch strict bullets (-, *) AND Mistral's weird bullet (•)
            elif s.startswith("- ") or s.startswith("* ") or s.startswith("•"):
                # Strip the bullet character and any extra spaces
                content = s[1:].strip()
                html.append(f'<p style="margin:2px 0 2px 16px;">&#8226;&nbsp;{content}</p>')

            # Catch horizontal rules
            elif s.startswith("---") or s.startswith("==="):
                html.append(f'<hr style="border:0;border-top:1px solid {C["border"]};margin:10px 0;">')

            else:
                # Handle inline bolding (**text**)
                formatted = s
                while "**" in formatted:
                    formatted = formatted.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
                html.append(f'<p style="margin:2px 0;">{formatted}</p>')

        # Close table if text ended while still inside one
        if in_table:
            html.append("</table>")

        full_html = f"""
        <html><body style="background:{C['bg']};
        color:{C['text_primary']};font-family:Segoe UI,sans-serif;
        font-size:13px;line-height:1.6;">
        {"".join(html)}
        </body></html>
        """
        self.setHtml(full_html)

    def append_chunk(self, chunk: str):
        cursor = self.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(chunk)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

# ═════════════════════════════════════════════════════════════════════════════
#  SectionHeader
# ═════════════════════════════════════════════════════════════════════════════

class SectionHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {C['text_primary']}; font-size: 16px; font-weight: 700;"
        )
        layout.addWidget(title_lbl)

        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setStyleSheet(
                f"color: {C['text_secondary']}; font-size: 11px;"
            )
            layout.addWidget(sub_lbl)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {C['border']}; margin-top: 4px;")
        layout.addWidget(line)
        
# ═════════════════════════════════════════════════════════════════════════════
#  ImpactChart — Vertical Bar Chart for Economic Data
# ═════════════════════════════════════════════════════════════════════════════

class ImpactChart(QWidget):
    """Draws a vertical bar chart comparing Gross, Net, and Lost Values."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(240)  # Taller height for a true graph look
        self.gross = 0.0
        self.net = 0.0
        self.loss = 0.0
        self.loss_pct = 0.0

    def update_chart(self, gross: float, net: float, loss: float, loss_pct: float):
        self.gross = gross
        self.net = net
        self.loss = loss
        self.loss_pct = loss_pct
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # 1. Draw Background Box
        painter.setBrush(QBrush(QColor(C["surface"])))
        painter.setPen(QPen(QColor(C["border"]), 1))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)

        if self.gross <= 0:
            painter.setPen(QColor(C["text_muted"]))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(QRect(0, 0, w, h), Qt.AlignCenter, "Run detection to generate production graph")
            return

        # 2. Setup Graph Canvas Area
        margin_left = 20
        margin_right = 20
        margin_top = 45
        margin_bottom = 35
        
        graph_w = w - margin_left - margin_right
        graph_h = h - margin_top - margin_bottom

        # Draw bottom X-axis line
        painter.setPen(QPen(QColor(C["border"]), 2))
        painter.drawLine(margin_left, h - margin_bottom, w - margin_right, h - margin_bottom)

        # 3. Data configuration
        bars = [
            {"label": "GROSS YIELD", "value": self.gross, "color": C["green_mid"]},
            {"label": "NET EXPECTED", "value": self.net, "color": C["green_light"]},
            {"label": "LOST YIELD", "value": self.loss, "color": C["red"]}
        ]
        
        max_val = self.gross if self.gross > 0 else 1
        scale = graph_h / max_val
        
        num_bars = len(bars)
        bar_spacing = graph_w / num_bars
        bar_width = min(80, int(bar_spacing * 0.5)) # Keep bars reasonably thick

        # 4. Draw the Bars and Text
        for i, bar in enumerate(bars):
            # Center of this specific bar's lane
            x_center = margin_left + (i * bar_spacing) + (bar_spacing / 2)
            x = int(x_center - (bar_width / 2))
            
            bar_h = int(bar["value"] * scale)
            y = h - margin_bottom - bar_h
            
            # Draw the colored bar
            painter.setBrush(QBrush(QColor(bar["color"])))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(x, y, bar_width, bar_h, 4, 4)
            
            # Square off the bottom of the bar so it sits flat on the X-axis
            if bar_h > 4:
                painter.drawRect(x, h - margin_bottom - 4, bar_width, 4)

            # Draw Labels (Underneath the X-axis)
            painter.setPen(QColor(C["text_secondary"]))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            text_rect = QRect(int(x_center - 75), h - margin_bottom + 10, 150, 20)
            painter.drawText(text_rect, Qt.AlignCenter, bar["label"])

            # Draw Values (Floating above the bars)
            val_text = f"${bar['value']:,.0f}"
            if bar["label"] == "LOST YIELD":
                val_text += f" ({self.loss_pct:.1f}%)"
                painter.setPen(QColor(C["red"]))
            else:
                painter.setPen(QColor(C["text_primary"]))
                
            painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
            val_rect = QRect(int(x_center - 100), y - 30, 200, 25)
            painter.drawText(val_rect, Qt.AlignCenter | Qt.AlignBottom, val_text)
