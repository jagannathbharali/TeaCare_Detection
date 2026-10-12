
C = {
    "bg":           "#0F1C14",   # deep forest black
    "surface":      "#162118",   # card backgrounds
    "surface2":     "#1E2E22",   # slightly lighter surface
    "border":       "#2A3D2E",   # subtle borders
    "green_deep":   "#2D6A4F",
    "green_mid":    "#40916C",
    "green_light":  "#74C69D",
    "green_pale":   "#B7E4C7",
    "cream":        "#F0EBD8",
    "amber":        "#F4A261",
    "red":          "#E63946",
    "text_primary": "#E8F0EB",
    "text_secondary":"#8FA896",
    "text_muted":   "#4D6B55",
    "accent":       "#74C69D",
    "severity_low":  "#74C69D",
    "severity_mod":  "#F4A261",
    "severity_high": "#E07B39",
    "severity_crit": "#E63946",
}


# ── Global stylesheet ─────────────────────────────────────────────────────────
GLOBAL_STYLE = f"""
/* ── Base ── */
QWidget {{
    background-color: {C['bg']};
    color: {C['text_primary']};
    font-family: 'Segoe UI', 'SF Pro Text', 'Ubuntu', sans-serif;
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background-color: {C['bg']};
}}

/* ── Scrollbars ── */
QScrollBar:vertical {{
    background: {C['surface']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C['green_deep']};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: {C['surface']};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {C['green_deep']};
    border-radius: 4px;
}}

/* ── Tabs ── */
QTabWidget::pane {{
    border: 1px solid {C['border']};
    background: {C['surface']};
    border-radius: 6px;
}}
QTabBar::tab {{
    background: {C['surface']};
    color: {C['text_secondary']};
    padding: 9px 22px;
    margin-right: 2px;
    border-radius: 4px 4px 0 0;
    font-size: 12px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background: {C['green_deep']};
    color: {C['text_primary']};
}}
QTabBar::tab:hover:!selected {{
    background: {C['surface2']};
    color: {C['green_light']};
}}

/* ── Buttons ── */
QPushButton {{
    background: {C['green_deep']};
    color: {C['cream']};
    border: none;
    border-radius: 6px;
    padding: 9px 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
}}
QPushButton:hover {{
    background: {C['green_mid']};
}}
QPushButton:pressed {{
    background: {C['green_light']};
    color: {C['bg']};
}}
QPushButton:disabled {{
    background: {C['border']};
    color: {C['text_muted']};
}}

/* ── Secondary button ── */
QPushButton[secondary="true"] {{
    background: transparent;
    border: 1px solid {C['green_deep']};
    color: {C['green_light']};
}}
QPushButton[secondary="true"]:hover {{
    background: {C['surface2']};
}}

/* ── Inputs ── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 5px;
    padding: 7px 10px;
    color: {C['text_primary']};
    selection-background-color: {C['green_deep']};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {C['green_mid']};
}}

/* ── Labels ── */
QLabel {{
    color: {C['text_primary']};
    background: transparent;
}}

/* ── Progress bar ── */
QProgressBar {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 5px;
    height: 14px;
    text-align: center;
    font-size: 10px;
    color: {C['text_secondary']};
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {C['green_deep']}, stop:1 {C['green_light']});
    border-radius: 4px;
}}

/* ── Combobox ── */
QComboBox {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 5px;
    padding: 6px 10px;
    color: {C['text_primary']};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {C['surface2']};
    border: 1px solid {C['border']};
    selection-background-color: {C['green_deep']};
    color: {C['text_primary']};
}}

/* ── Table ── */
QTableWidget {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    gridline-color: {C['border']};
    border-radius: 6px;
    alternate-background-color: {C['surface2']};
}}
QTableWidget::item {{
    padding: 5px 8px;
    color: {C['text_primary']};
}}
QTableWidget::item:selected {{
    background: {C['green_deep']};
}}
QHeaderView::section {{
    background: {C['surface2']};
    color: {C['green_light']};
    border: none;
    border-bottom: 1px solid {C['border']};
    padding: 7px 8px;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}

/* ── Splitter ── */
QSplitter::handle {{
    background: {C['border']};
}}

/* ── Tooltip ── */
QToolTip {{
    background: {C['surface2']};
    border: 1px solid {C['green_mid']};
    color: {C['text_primary']};
    padding: 4px 8px;
    border-radius: 4px;
}}

/* ── GroupBox ── */
QGroupBox {{
    border: 1px solid {C['border']};
    border-radius: 6px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    color: {C['green_light']};
    font-size: 11px;
    letter-spacing: 0.5px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    background: {C['bg']};
}}

/* ── Slider ── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {C['border']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px;
    background: {C['green_light']};
    border-radius: 7px;
    margin: -5px 0;
}}
QSlider::sub-page:horizontal {{
    background: {C['green_mid']};
    border-radius: 2px;
}}
"""


# ── Card style helper ─────────────────────────────────────────────────────────
def card_style(radius: int = 8, border: bool = True) -> str:
    b = f"border: 1px solid {C['border']};" if border else "border: none;"
    return f"""
        background-color: {C['surface']};
        {b}
        border-radius: {radius}px;
        padding: 12px;
    """


def severity_colour(label: str) -> str:
    return {
        "Low":      C["severity_low"],
        "Moderate": C["severity_mod"],
        "High":     C["severity_high"],
        "Critical": C["severity_crit"],
    }.get(label, C["text_secondary"])