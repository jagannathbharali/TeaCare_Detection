"""
utils/report_exporter.py
────────────────────────
Export detection results + economic analysis to PDF and JSON.
Uses ReportLab for rich PDF layout and matplotlib for charts.
"""

from __future__ import annotations

import json
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak,
)
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from loguru import logger

# Palette
_GREEN  = colors.HexColor("#2D6A4F")
_LIME   = colors.HexColor("#74C69D")
_AMBER  = colors.HexColor("#F4A261")
_RED    = colors.HexColor("#E63946")
_CREAM  = colors.HexColor("#F8F9F0")
_DARK   = colors.HexColor("#1B2A22")


# ═════════════════════════════════════════════════════════════════════════════
#  Matplotlib chart helpers
# ═════════════════════════════════════════════════════════════════════════════

def _disease_pie_chart(disease_pct: dict[str, float]) -> io.BytesIO:
    palette = ["#2D6A4F","#74C69D","#F4A261","#E63946",
               "#A8DADC","#457B9D","#E9C46A","#F77F00"]
    labels  = list(disease_pct.keys())
    values  = list(disease_pct.values())

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels,
        colors=palette[:len(labels)],
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Disease Distribution", fontsize=11, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _confidence_bar_chart(results_list: list[dict]) -> io.BytesIO:
    """Bar chart of confidence scores for first 20 images."""
    subset   = results_list[:20]
    labels   = [Path(r["image_path"]).stem[:12] for r in subset]
    confs    = [r["confidence"] * 100 for r in subset]
    diseases = [r["top_disease"] for r in subset]

    cmap = {"Healthy": "#2D6A4F"}
    bar_colors = [cmap.get(d, "#E63946") for d in diseases]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(range(len(labels)), confs, color=bar_colors, edgecolor="white", width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 105)
    ax.axhline(70, color="orange", linestyle="--", linewidth=0.8, label="70% threshold")
    ax.set_title("Per-Image Detection Confidence", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ═════════════════════════════════════════════════════════════════════════════
#  PDF Builder
# ═════════════════════════════════════════════════════════════════════════════

def export_pdf(
    output_path: str | Path,
    batch_summary: dict,
    economic_estimate,
    economic_report_md: str,
    results_list: list[dict],
    sample_images: list[np.ndarray] | None = None,
) -> Path:
    """
    Build a complete PDF report.

    Parameters
    ----------
    output_path        : destination .pdf file
    batch_summary      : BatchResult.summary_dict()
    economic_estimate  : EconomicEstimate instance
    economic_report_md : markdown text from Mistral
    results_list       : list of dicts (each SingleResult serialised)
    sample_images      : up to 6 BGR numpy arrays for preview grid
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc    = SimpleDocTemplate(str(output_path), pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # ── Custom styles ──────────────────────────────────────────────────────
    h1 = ParagraphStyle("H1", fontSize=22, textColor=_GREEN,
                        fontName="Helvetica-Bold", spaceAfter=6, leading=26)
    h2 = ParagraphStyle("H2", fontSize=14, textColor=_GREEN,
                        fontName="Helvetica-Bold", spaceAfter=4,
                        spaceBefore=12, leading=18)
    h3 = ParagraphStyle("H3", fontSize=11, textColor=_DARK,
                        fontName="Helvetica-Bold", spaceAfter=3,
                        spaceBefore=6, leading=14)
    body = ParagraphStyle("Body", fontSize=9, textColor=_DARK,
                          fontName="Helvetica", leading=14, spaceAfter=4)
    mono = ParagraphStyle("Mono", fontSize=8, textColor=_DARK,
                          fontName="Courier", leading=12)
    sub  = ParagraphStyle("Sub", fontSize=8, textColor=colors.grey,
                          fontName="Helvetica", leading=10)

    # ══════════════════════════════════════════════════════════
    #  COVER
    # ══════════════════════════════════════════════════════════
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("🍃 TeaGuard AI", h1))
    story.append(Paragraph("Tea Leaf Disease Detection & Economic Impact Report", h2))
    story.append(HRFlowable(width="100%", thickness=2, color=_GREEN))
    story.append(Spacer(1, 4*mm))
    ts = datetime.now().strftime("%d %B %Y — %H:%M")
    story.append(Paragraph(f"Generated: {ts}", sub))
    story.append(Spacer(1, 6*mm))

    # ── Key stats banner ──────────────────────────────────────
    sym   = "$"
    sev   = economic_estimate.severity_label
    sev_c = {
        "Low": colors.green, "Moderate": _AMBER,
        "High": colors.orange, "Critical": _RED
    }.get(sev, colors.grey)

    banner_data = [
        ["Images Analysed", "Diseased", "Dominant Disease",
         "Estimated Loss", "Severity"],
        [
            str(batch_summary["total_images"]),
            f"{batch_summary['diseased_pct']:.1f}%",
            batch_summary["dominant_disease"],
            f"{sym}{economic_estimate.estimated_loss_usd:,.0f}",
            sev,
        ],
    ]
    banner_tbl = Table(banner_data, colWidths=[3.2*cm]*5)
    banner_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), _GREEN),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 9),
        ("BACKGROUND", (0,1), (-1,1), _CREAM),
        ("FONTNAME",   (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,1), (-1,1), 11),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,1), [_CREAM]),
        ("TEXTCOLOR",  (4,1), (4,1), sev_c),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.white),
        ("ROWHEIGHT",  (0,0), (-1,-1), 18),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 8*mm))

    # ══════════════════════════════════════════════════════════
    #  DETECTION RESULTS
    # ══════════════════════════════════════════════════════════
    story.append(Paragraph("1. Detection Results", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_LIME))
    story.append(Spacer(1, 3*mm))

    # Charts side by side
    pie_buf  = _disease_pie_chart(batch_summary["disease_percentages"])
    conf_buf = _confidence_bar_chart(results_list)

    chart_data = [[
        RLImage(pie_buf,  width=7*cm, height=5.5*cm),
        RLImage(conf_buf, width=10*cm, height=5.5*cm),
    ]]
    chart_tbl = Table(chart_data, colWidths=[8*cm, 11*cm])
    chart_tbl.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(chart_tbl)
    story.append(Spacer(1, 4*mm))

    # Disease breakdown table
    story.append(Paragraph("Disease Breakdown", h3))
    bd_header = ["Disease", "Count", "% of Batch", "Severity"]
    bd_rows   = []
    from pathlib import Path as _P
    import yaml as _yaml
    _cfg = _yaml.safe_load(open(_P(__file__).parent.parent / "config.yaml"))
    sev_map = _cfg["diseases"]["severity"]
    loss_map = _cfg["diseases"]["crop_loss_pct"]

    for disease, pct in batch_summary["disease_percentages"].items():
        count    = batch_summary["disease_counts"][disease]
        severity = ["—","Low","Moderate","High","Critical"][min(sev_map.get(disease,0),4)]
        bd_rows.append([disease, str(count), f"{pct:.1f}%", severity])

    bd_table = Table([bd_header] + bd_rows, colWidths=[6*cm, 3*cm, 4*cm, 4*cm])
    bd_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), _GREEN),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, _CREAM]),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("ALIGN",      (1,0), (-1,-1), "CENTER"),
        ("ROWHEIGHT",  (0,0), (-1,-1), 16),
    ]))
    story.append(bd_table)
    story.append(Spacer(1, 3*mm))

    # Per-image results table (first 30)
    story.append(Paragraph("Per-Image Results (first 30)", h3))
    pi_header = ["Image", "Predicted Disease", "Confidence", "Loss Est."]
    pi_rows = []
    for r in results_list[:30]:
        pi_rows.append([
            Path(r["image_path"]).name[:30],
            r["top_disease"],
            f"{r['confidence']*100:.1f}%",
            f"{r['crop_loss_pct']:.0f}%",
        ])
    pi_table = Table([pi_header] + pi_rows,
                     colWidths=[6*cm, 5*cm, 3.5*cm, 3.5*cm])
    pi_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), _GREEN),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, _CREAM]),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("ALIGN",      (2,0), (-1,-1), "CENTER"),
        ("ROWHEIGHT",  (0,0), (-1,-1), 14),
    ]))
    story.append(pi_table)

    # ══════════════════════════════════════════════════════════
    #  ECONOMIC IMPACT
    # ══════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("2. Economic Impact Analysis", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_LIME))
    story.append(Spacer(1, 3*mm))

    # Financial summary table
    fin_data = [
        ["Metric", "Value"],
        ["Gross Crop Value", f"{sym}{economic_estimate.gross_crop_value_usd:,.2f}"],
        ["Estimated Loss",   f"{sym}{economic_estimate.estimated_loss_usd:,.2f} "
                             f"({economic_estimate.batch_loss_pct:.1f}%)"],
        ["Net Expected Value", f"{sym}{economic_estimate.net_value_usd:,.2f}"],
        ["Severity Level",   economic_estimate.severity_label],
    ]
    fin_table = Table(fin_data, colWidths=[8*cm, 9*cm])
    fin_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), _GREEN),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, _CREAM]),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("FONTNAME",   (0,-1), (-1,-1), "Helvetica-Bold"),
        ("ALIGN",      (1,0), (1,-1), "RIGHT"),
        ("ROWHEIGHT",  (0,0), (-1,-1), 18),
    ]))
    story.append(fin_table)
    story.append(Spacer(1, 5*mm))

    # ── Mistral report (convert markdown to ReportLab paragraphs) ─────────
    story.append(Paragraph("AI-Generated Economic Analysis (Mistral)", h3))
    story.append(Spacer(1, 2*mm))

    for line in economic_report_md.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 2*mm))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], h3))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:], h2))
        elif line.startswith("- ") or line.startswith("* "):
            story.append(Paragraph(f"• {line[2:]}", body))
        elif line.startswith("|"):
            pass   # skip raw markdown tables in PDF
        else:
            story.append(Paragraph(line, body))

    # ── Footer note ───────────────────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_LIME))
    story.append(Paragraph(
        "Generated by TeaGuard AI — Powered by YOLOv8 + Mistral. "
        "Economic estimates use estate-level averages; consult an agronomist "
        "for precise field management decisions.",
        sub,
    ))

    doc.build(story)
    logger.success(f"PDF exported: {output_path}")
    return output_path


# ═════════════════════════════════════════════════════════════════════════════
#  JSON Export
# ═════════════════════════════════════════════════════════════════════════════

def export_json(
    output_path: str | Path,
    batch_summary: dict,
    economic_estimate,
    results_list: list[dict],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "app": "TeaGuard AI v1.0",
        },
        "batch_summary": batch_summary,
        "economic_estimate": {
            "gross_crop_value_usd":  economic_estimate.gross_crop_value_usd,
            "estimated_loss_usd":    economic_estimate.estimated_loss_usd,
            "net_value_usd":         economic_estimate.net_value_usd,
            "batch_loss_pct":        economic_estimate.batch_loss_pct,
            "severity_label":        economic_estimate.severity_label,
            "dominant_disease":      economic_estimate.dominant_disease,
        },
        "per_image_results": results_list,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.success(f"JSON exported: {output_path}")
    return output_path