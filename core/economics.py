
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml
import requests
from loguru import logger

try:
    import ollama as _ollama_client
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False


# ── Config ────────────────────────────────────────────────────────────────────
def _load_cfg():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

CFG = _load_cfg()

#  Economic Calculator

@dataclass
class EconomicEstimate:
    """Quantitative economic summary for one batch analysis."""
    total_batch_images:   int
    diseased_images:      int
    diseased_pct:         float
    dominant_disease:     str
    batch_loss_pct:       float
    gross_crop_value_usd: float
    estimated_loss_usd:   float
    net_value_usd:        float
    disease_breakdown:    dict[str, float]
    severity_label:       str


def compute_economic_estimate(batch_summary: dict) -> EconomicEstimate:
    """
    Compute monetary estimates from batch detection summary.
    batch_summary is the dict returned by BatchResult.summary_dict().
    """
    eco = CFG["economics"]

    yield_kg    = eco["avg_yield_kg_per_hectare"]
    price_usd   = eco["avg_price_per_kg_usd"]
    hectares    = eco["avg_estate_hectares"]

    gross_value = yield_kg * price_usd * hectares
    loss_pct    = batch_summary["batch_loss_pct"]
    loss_usd    = gross_value * (loss_pct / 100.0)
    net_value   = gross_value - loss_usd

    diseased_n  = batch_summary["total_images"] - \
                  batch_summary["disease_counts"].get("Healthy", 0)

    if loss_pct < 10:
        severity = "Low"
    elif loss_pct < 25:
        severity = "Moderate"
    elif loss_pct < 45:
        severity = "High"
    else:
        severity = "Critical"

    return EconomicEstimate(
        total_batch_images   = batch_summary["total_images"],
        diseased_images      = diseased_n,
        diseased_pct         = batch_summary["diseased_pct"],
        dominant_disease     = batch_summary["dominant_disease"],
        batch_loss_pct       = loss_pct,
        gross_crop_value_usd = round(gross_value, 2),
        estimated_loss_usd   = round(loss_usd, 2),
        net_value_usd        = round(net_value, 2),
        disease_breakdown    = batch_summary["disease_percentages"],
        severity_label       = severity,
    )

#  Mistral LLM Report Generator

_SYSTEM_PROMPT = """\
You are an expert agricultural economist and tea industry analyst.
You write clear, detailed economic impact reports for tea estate managers.
Your reports are factual, structured, and actionable.
Use markdown headings (##) to organise sections.
Use standard markdown bullet points (-) for all lists. Do not use • symbols.
Always format tables using strict Markdown syntax with pipe (|) characters.
Do not use generic filler text — every sentence must be specific to
the disease, severity, and numbers provided.
"""

_REPORT_TEMPLATE = """\
Analyse the following tea leaf disease detection results for a tea estate
and generate a comprehensive Economic Impact Report.

## Detection Data
- Total images analysed: {total_images}
- Diseased images: {diseased_images} ({diseased_pct}%)
- Disease breakdown: {disease_breakdown}
- Dominant disease: {dominant_disease}
- Severity level: {severity_label}

## Financial Baseline
- Gross crop value (before disease): ${gross_value:,.0f} USD
- Estimated crop loss: ${loss_usd:,.0f} USD ({loss_pct:.1f}%)
- Net expected value: ${net_value:,.0f} USD

---

Please write a detailed Economic Impact Report with the following sections:

## 1. Executive Summary
A 3-4 sentence overview of the situation, the dominant disease,
and the financial exposure.

## 2. Disease Profile & Agronomic Impact
Explain what {dominant_disease} is, how it spreads in tea gardens,
and which parts of the plant it affects. Include typical yield-loss
ranges from literature.

## 3. Supply & Demand Analysis
- How does a {loss_pct:.1f}% crop loss affect the local supply curve
  for this estate's tea?
- What is the likely price effect in the local auction market?
- Are there downstream effects on blending factories or exporters?

## 4. Local Market Structure Effects
- Impact on estate workers (labour demand, seasonal picking)
- Effect on small-holder tea farmers in the same region
- Potential for market concentration if multiple estates are affected

## 5. Crop Value Depreciation Breakdown
- Which grades of tea suffer the most from {dominant_disease}?
- How does post-harvest processing cost change when infected leaves
  enter the batch?
- Provide a table using exact Markdown format like this:
| Grade | Normal Value | Diseased Value | Depreciation % |
|---|---|---|---|
| Orthodox | ... | ... | ... |
| CTC | ... | ... | ... |

## 6. Risk Escalation Scenarios
Describe what happens if disease coverage grows from the current
{diseased_pct:.0f}% to 50%, 75%, and 100% of the estate.

## 7. Recommended Actions & Cost-Benefit
List 4-6 specific interventions, their estimated cost per hectare,
and the expected crop value saved per intervention.

## 8. Conclusion
A concise closing paragraph summarising urgency, recommended
priority actions, and expected recovery timeline.
"""

def _call_ollama_http(prompt: str) -> str:
    llm_cfg = CFG["llm"]
    url     = f"{llm_cfg['host']}/api/generate"
    payload = {
        "model":   llm_cfg["model"],
        "prompt":  prompt,
        "stream":  False,
        "options": {"num_predict": llm_cfg["max_tokens"]},
    }
    resp = requests.post(url, json=payload, timeout=llm_cfg["timeout"])
    resp.raise_for_status()
    return resp.json().get("response", "")


def generate_economic_report(
    estimate: EconomicEstimate,
    *,
    stream_callback=None,
) -> str:
    llm_cfg = CFG["llm"]

    user_prompt = _REPORT_TEMPLATE.format(
        total_images      = estimate.total_batch_images,
        diseased_images   = estimate.diseased_images,
        diseased_pct      = estimate.diseased_pct,
        disease_breakdown = json.dumps(estimate.disease_breakdown, indent=2),
        dominant_disease  = estimate.dominant_disease,
        severity_label    = estimate.severity_label,
        gross_value       = estimate.gross_crop_value_usd,
        loss_usd          = estimate.estimated_loss_usd,
        loss_pct          = estimate.batch_loss_pct,
        net_value         = estimate.net_value_usd,
    )

    full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_prompt}"

    logger.info(f"Calling Mistral via Ollama ({llm_cfg['host']}) ...")

    # Try streaming via ollama Python client
    if _HAS_OLLAMA and stream_callback:
        try:
            report_chunks = []
            for chunk in _ollama_client.generate(
                model=llm_cfg["model"],
                prompt=full_prompt,
                stream=True,
            ):
                text = chunk.get("response", "")
                if text:
                    report_chunks.append(text)
                    stream_callback(text)
                if chunk.get("done"):
                    break
            return "".join(report_chunks)
        except Exception as exc:
            logger.warning(f"Ollama streaming failed: {exc}. Retrying via HTTP.")

    # Fallback: plain HTTP call
    try:
        report = _call_ollama_http(full_prompt)
        if stream_callback:
            stream_callback(report)
        return report
    except requests.ConnectionError:
        msg = (
            "Could not connect to Ollama. Make sure Ollama is running:\n"
            "    ollama serve\n"
            "And the Mistral model is downloaded:\n"
            "    ollama pull mistral\n\n"
            "── Offline Fallback Report ──\n\n"
            + _offline_fallback_report(estimate)
        )
        logger.error("Ollama not reachable — returning offline fallback report.")
        if stream_callback:
            stream_callback(msg)
        return msg
    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        return f"Error generating report: {exc}"


def _offline_fallback_report(est: EconomicEstimate) -> str:
    """Template-only report used when Mistral is unavailable."""
    sym = CFG["economics"]["currency_symbol"]
    return f"""
## Economic Impact Summary (Offline Mode)

**Estate Analysis**
| Metric | Value |
|---|---|
| Images analysed | {est.total_batch_images} |
| Diseased images | {est.diseased_images} ({est.diseased_pct:.1f}%) |
| Dominant disease | {est.dominant_disease} |
| Severity | {est.severity_label} |
| Gross crop value | {sym}{est.gross_crop_value_usd:,.0f} |
| Estimated loss | {sym}{est.estimated_loss_usd:,.0f} ({est.batch_loss_pct:.1f}%) |
| Net value | {sym}{est.net_value_usd:,.0f} |

**Disease Breakdown**
{chr(10).join(f'- {k}: {v:.1f}%' for k, v in est.disease_breakdown.items())}

> Ollama + Mistral required for full AI report.
> Run: ollama serve
> Then: ollama pull mistral
"""
