#!/usr/bin/env python3
# generate_report.py
#
# Reads all JSON result files from a results directory and generates
# a single self-contained HTML report.
#
# Usage:
#   python generate_report.py                          # reads ./results/, writes ./results/report.html
#   python generate_report.py --results-dir results/  # explicit dir
#   python generate_report.py --output my_report.html # custom output path

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> list[dict]:
    files = sorted(results_dir.glob("*.json"))
    results = []
    for f in files:
        if f.name == "report.json":
            continue
        try:
            with f.open(encoding="utf-8") as fh:
                results.append(json.load(fh))
        except Exception as e:
            print(f"Warning: could not read {f.name}: {e}")
    return results


def status_badge(status: str) -> str:
    colors = {
        "excellent": ("#d1fae5", "#065f46", "Excellent"),
        "acceptable": ("#dbeafe", "#1e40af", "Acceptable"),
        "failed":     ("#fee2e2", "#991b1b", "Failed"),
        "no_face":    ("#fef3c7", "#92400e", "No Face"),
        "pass":       ("#d1fae5", "#065f46", "Pass"),
        "marginal":   ("#fef3c7", "#92400e", "Marginal"),
        "fail":       ("#fee2e2", "#991b1b", "Fail"),
        "error":      ("#f3f4f6", "#6b7280", "Error"),
    }
    bg, fg, label = colors.get(status, ("#f3f4f6", "#6b7280", status.title()))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:600;">{label}</span>'
    )


def fmt_score(score) -> str:
    if score is None:
        return '<span style="color:#9ca3af">—</span>'
    return f"{score:.4f}"


def image_tag(path_str: str, alt: str = "") -> str:
    """Return an <img> tag using a file:// URL, or a placeholder if path is empty."""
    if not path_str or not path_str.strip():
        return '<span style="color:#9ca3af;font-size:12px">No image</span>'
    p = Path(path_str)
    # Use file:// so the browser can load local images when the report is opened locally
    url = p.as_uri()
    return (
        f'<img src="{url}" alt="{alt}" '
        f'style="width:90px;height:90px;object-fit:cover;border-radius:6px;'
        f'border:1px solid #e5e7eb;" '
        f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'inline\'">'
        f'<span style="display:none;color:#9ca3af;font-size:11px">Image unavailable</span>'
    )


def compute_summary(results: list[dict]) -> dict:
    face_scores = [r["face_similarity"]["score"] for r in results if r["face_similarity"]["score"] is not None]
    clip_scores = [r["prompt_adherence"]["score"] for r in results if r["prompt_adherence"]["score"] is not None]

    face_statuses  = [r["face_similarity"]["status"] for r in results]
    clip_statuses  = [r["prompt_adherence"]["status"] for r in results]
    qual_statuses  = [r["quality"]["overall_status"] for r in results if r.get("quality")]
    artf_statuses  = [r["artifact"]["overall_status"] for r in results if r.get("artifact")]
    safe_statuses  = [r["safety"]["overall_status"] for r in results if r.get("safety")]
    style_statuses = [r["style"]["overall_status"] for r in results if r.get("style")]
    style_labels   = [r["style"]["style_label"] for r in results if r.get("style")]

    return {
        "total": len(results),
        "face_avg": sum(face_scores) / len(face_scores) if face_scores else None,
        "clip_avg": sum(clip_scores) / len(clip_scores) if clip_scores else None,
        "face_excellent":  face_statuses.count("excellent"),
        "face_acceptable": face_statuses.count("acceptable"),
        "face_failed":     face_statuses.count("failed"),
        "face_no_face":    face_statuses.count("no_face"),
        "face_errors":     face_statuses.count("error"),
        "clip_pass":       clip_statuses.count("pass"),
        "clip_marginal":   clip_statuses.count("marginal"),
        "clip_fail":       clip_statuses.count("fail"),
        "clip_errors":     clip_statuses.count("error"),
        "qual_pass":       qual_statuses.count("pass"),
        "qual_fail":       qual_statuses.count("fail"),
        "qual_errors":     qual_statuses.count("error"),
        "artf_pass":       artf_statuses.count("pass"),
        "artf_flagged":    artf_statuses.count("flagged"),
        "artf_errors":     artf_statuses.count("error"),
        "safe_safe":       safe_statuses.count("safe"),
        "safe_flagged":    safe_statuses.count("flagged"),
        "safe_errors":     safe_statuses.count("error"),
        "style_pass":      style_statuses.count("pass"),
        "style_mismatch":  style_statuses.count("mismatch"),
        "style_errors":    style_statuses.count("error"),
        "style_photorealistic": style_labels.count("photorealistic"),
        "style_anime":     style_labels.count("anime"),
        "style_cartoon":   style_labels.count("cartoon"),
        "style_other":     sum(1 for l in style_labels if l not in ("photorealistic", "anime", "cartoon")),
        "llms": sorted(set(r.get("llm_used") or "Unknown" for r in results)),
    }


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(results: list[dict], generated_at: str) -> str:
    s = compute_summary(results)

    face_avg_str = f"{s['face_avg']:.4f}" if s["face_avg"] is not None else "N/A"
    clip_avg_str = f"{s['clip_avg']:.4f}" if s["clip_avg"] is not None else "N/A"
    llm_str = ", ".join(s["llms"]) if s["llms"] else "—"

    # Build rows
    rows_html = ""
    for r in results:
        face  = r.get("face_similarity", {})
        clip  = r.get("prompt_adherence", {})
        qual  = r.get("quality") or {}
        artf  = r.get("artifact") or {}
        safe  = r.get("safety") or {}
        style = r.get("style") or {}
        ref_img = r.get("identity_image", "")
        gen_img = r.get("generated_image", "")
        prompt  = r.get("prompt", "")[:120] + ("…" if len(r.get("prompt","")) > 120 else "")
        llm     = r.get("llm_used") or "—"
        row_id  = r.get("id", "?")
        evaluated_at = r.get("evaluated_at", "")[:19].replace("T", " ") if r.get("evaluated_at") else "—"

        face_error = face.get("error") or ""
        error_html = f'<div style="color:#dc2626;font-size:11px;margin-top:4px">⚠ {face_error}</div>' if face_error else ""

        # Quality cell
        if qual:
            blur_str = f"Blur: {qual.get('blur_score','—')}"
            res_str  = f"{qual.get('resolution_width','?')}×{qual.get('resolution_height','?')}"
            qual_html = f'{fmt_score(None) if qual.get("overall_status")=="error" else ""}<div style="font-size:11px;color:#6b7280">{blur_str}<br>{res_str}</div><div style="margin-top:4px">{status_badge(qual.get("overall_status","error"))}</div>'
        else:
            qual_html = status_badge("error")

        # Artifact cell
        if artf:
            flagged = artf.get("flagged_categories") or []
            flag_str = ", ".join(flagged) if flagged else "none"
            artf_html = f'<div style="font-size:11px;color:#6b7280;margin-bottom:4px">Flagged: {flag_str}</div>{status_badge(artf.get("overall_status","error"))}'
        else:
            artf_html = status_badge("error")

        # Safety cell
        if safe:
            flagged_s  = safe.get("flagged_categories") or []
            flag_str_s = ", ".join(flagged_s) if flagged_s else "none"
            safe_html  = f'<div style="font-size:11px;color:#6b7280;margin-bottom:4px">Flagged: {flag_str_s}</div>{status_badge(safe.get("overall_status","error"))}'
        else:
            safe_html = status_badge("error")

        # Style cell
        if style:
            style_label = style.get("style_label", "unknown")
            style_match = style.get("style_match")
            style_status = style.get("overall_status", "error")
            match_str = ""
            if style_match is False:
                match_str = '<div style="font-size:10px;color:#dc2626;margin-top:2px">⚠ style mismatch</div>'
            elif style_match is True:
                match_str = '<div style="font-size:10px;color:#059669;margin-top:2px">✓ style match</div>'
            style_html = (
                f'<div style="font-size:11px;color:#6b7280;margin-bottom:4px">{style_label}</div>'
                f'{status_badge(style_status)}'
                f'{match_str}'
            )
        else:
            style_html = status_badge("error")

        rows_html += f"""
        <tr>
          <td style="font-weight:600;white-space:nowrap">{row_id}</td>
          <td style="max-width:180px;font-size:12px;color:#374151">{prompt}</td>
          <td style="text-align:center">{image_tag(ref_img, "reference")}</td>
          <td style="text-align:center">{image_tag(gen_img, "generated")}</td>
          <td style="text-align:center;font-size:12px">{llm}</td>
          <td style="text-align:center">
            {fmt_score(face.get("score"))}<br>
            <div style="margin-top:4px">{status_badge(face.get("status","error"))}</div>
            {error_html}
          </td>
          <td style="text-align:center">
            {fmt_score(clip.get("score"))}<br>
            <div style="margin-top:4px">{status_badge(clip.get("status","error"))}</div>
            {f'<div style="margin-top:4px;font-size:10px;color:#6b7280">⚡ {clip.get("chunks_used")} chunks · ~{clip.get("token_count")} tokens</div>' if clip.get("chunks_used", 1) > 1 else ""}
          </td>
          <td style="text-align:center">{qual_html}</td>
          <td style="text-align:center">{artf_html}</td>
          <td style="text-align:center">{safe_html}</td>
          <td style="text-align:center">{style_html}</td>
          <td style="font-size:11px;color:#6b7280;white-space:nowrap">{evaluated_at}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BLAST Evaluation Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f9fafb; color: #111827; }}
    .header {{ background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%); color: white; padding: 32px 40px; }}
    .header h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }}
    .header p {{ margin-top: 6px; opacity: 0.8; font-size: 14px; }}
    .body {{ padding: 32px 40px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 32px; }}
    .card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e5e7eb; }}
    .card .label {{ font-size: 12px; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
    .card .value {{ font-size: 28px; font-weight: 700; color: #1e3a5f; margin-top: 4px; }}
    .card .sub {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
    .section-title {{ font-size: 16px; font-weight: 600; color: #374151; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #e5e7eb; }}
    .breakdown {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 32px; }}
    .breakdown-block {{ background: white; border-radius: 12px; padding: 20px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e5e7eb; min-width: 220px; }}
    .breakdown-block h3 {{ font-size: 13px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}
    .breakdown-row {{ display: flex; justify-content: space-between; align-items: center; padding: 4px 0; font-size: 13px; }}
    .breakdown-row .count {{ font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e5e7eb; }}
    thead {{ background: #f3f4f6; }}
    th {{ padding: 12px 14px; text-align: left; font-size: 12px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.4px; white-space: nowrap; }}
    td {{ padding: 14px; border-top: 1px solid #f3f4f6; vertical-align: middle; font-size: 13px; }}
    tr:hover td {{ background: #f9fafb; }}
    .footer {{ text-align: center; padding: 24px; font-size: 12px; color: #9ca3af; }}
  </style>
</head>
<body>

<div class="header">
  <h1>BLAST Evaluation Report</h1>
  <p>Identity-Preserving Image Generation · Generated {generated_at} · {s['total']} rows · Models: {llm_str}</p>
</div>

<div class="body">

  <!-- Summary cards -->
  <div class="summary-grid">
    <div class="card">
      <div class="label">Total Rows</div>
      <div class="value">{s['total']}</div>
      <div class="sub">{s['total'] - s['face_errors']} evaluated, {s['face_errors']} errors</div>
    </div>
    <div class="card">
      <div class="label">Avg Face Score</div>
      <div class="value">{face_avg_str}</div>
      <div class="sub">ArcFace cosine similarity</div>
    </div>
    <div class="card">
      <div class="label">Avg CLIP Score</div>
      <div class="value">{clip_avg_str}</div>
      <div class="sub">Prompt adherence</div>
    </div>
    <div class="card">
      <div class="label">Face Pass Rate</div>
      <div class="value">{s['face_excellent'] + s['face_acceptable']}/{s['total']}</div>
      <div class="sub">Excellent + Acceptable</div>
    </div>
    <div class="card">
      <div class="label">CLIP Pass Rate</div>
      <div class="value">{s['clip_pass']}/{s['total']}</div>
      <div class="sub">Score ≥ 0.20</div>
    </div>
    <div class="card">
      <div class="label">Quality Pass</div>
      <div class="value">{s['qual_pass']}/{s['total']}</div>
      <div class="sub">Sharp + good resolution</div>
    </div>
    <div class="card">
      <div class="label">Artifacts Flagged</div>
      <div class="value">{s['artf_flagged']}/{s['total']}</div>
      <div class="sub">BLIP VQA</div>
    </div>
    <div class="card">
      <div class="label">Safety Flagged</div>
      <div class="value">{s['safe_flagged']}/{s['total']}</div>
      <div class="sub">NSFW / violence / harmful</div>
    </div>
    <div class="card">
      <div class="label">Style Mismatch</div>
      <div class="value">{s['style_mismatch']}/{s['total']}</div>
      <div class="sub">Realism vs prompt expectation</div>
    </div>
  </div>

  <!-- Breakdown -->
  <div class="breakdown">
    <div class="breakdown-block">
      <h3>Face Similarity</h3>
      <div class="breakdown-row"><span>{status_badge("excellent")}</span><span class="count">{s['face_excellent']}</span></div>
      <div class="breakdown-row"><span>{status_badge("acceptable")}</span><span class="count">{s['face_acceptable']}</span></div>
      <div class="breakdown-row"><span>{status_badge("failed")}</span><span class="count">{s['face_failed']}</span></div>
      <div class="breakdown-row"><span>{status_badge("no_face")}</span><span class="count">{s['face_no_face']}</span></div>
      <div class="breakdown-row"><span>{status_badge("error")}</span><span class="count">{s['face_errors']}</span></div>
    </div>
    <div class="breakdown-block">
      <h3>Prompt Adherence</h3>
      <div class="breakdown-row"><span>{status_badge("pass")}</span><span class="count">{s['clip_pass']}</span></div>
      <div class="breakdown-row"><span>{status_badge("marginal")}</span><span class="count">{s['clip_marginal']}</span></div>
      <div class="breakdown-row"><span>{status_badge("fail")}</span><span class="count">{s['clip_fail']}</span></div>
      <div class="breakdown-row"><span>{status_badge("error")}</span><span class="count">{s['clip_errors']}</span></div>
    </div>
    <div class="breakdown-block">
      <h3>Quality</h3>
      <div class="breakdown-row"><span>{status_badge("pass")}</span><span class="count">{s['qual_pass']}</span></div>
      <div class="breakdown-row"><span>{status_badge("fail")}</span><span class="count">{s['qual_fail']}</span></div>
      <div class="breakdown-row"><span>{status_badge("error")}</span><span class="count">{s['qual_errors']}</span></div>
    </div>
    <div class="breakdown-block">
      <h3>Artifacts</h3>
      <div class="breakdown-row"><span>{status_badge("pass")}</span><span class="count">{s['artf_pass']}</span></div>
      <div class="breakdown-row"><span style="background:#fef3c7;color:#92400e;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">Flagged</span><span class="count">{s['artf_flagged']}</span></div>
      <div class="breakdown-row"><span>{status_badge("error")}</span><span class="count">{s['artf_errors']}</span></div>
    </div>
    <div class="breakdown-block">
      <h3>Safety</h3>
      <div class="breakdown-row"><span style="background:#d1fae5;color:#065f46;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">Safe</span><span class="count">{s['safe_safe']}</span></div>
      <div class="breakdown-row"><span style="background:#fef3c7;color:#92400e;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">Flagged</span><span class="count">{s['safe_flagged']}</span></div>
      <div class="breakdown-row"><span>{status_badge("error")}</span><span class="count">{s['safe_errors']}</span></div>
    </div>
    <div class="breakdown-block">
      <h3>Style</h3>
      <div class="breakdown-row"><span>{status_badge("pass")}</span><span class="count">{s['style_pass']}</span></div>
      <div class="breakdown-row"><span style="background:#fee2e2;color:#991b1b;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">Mismatch</span><span class="count">{s['style_mismatch']}</span></div>
      <div class="breakdown-row"><span>{status_badge("error")}</span><span class="count">{s['style_errors']}</span></div>
      <div style="margin-top:8px;border-top:1px solid #f3f4f6;padding-top:8px">
        <div class="breakdown-row"><span style="font-size:11px;color:#6b7280">Photorealistic</span><span class="count" style="font-size:12px">{s['style_photorealistic']}</span></div>
        <div class="breakdown-row"><span style="font-size:11px;color:#6b7280">Anime</span><span class="count" style="font-size:12px">{s['style_anime']}</span></div>
        <div class="breakdown-row"><span style="font-size:11px;color:#6b7280">Cartoon/Other</span><span class="count" style="font-size:12px">{s['style_cartoon'] + s['style_other']}</span></div>
      </div>
    </div>
    <div class="breakdown-block">
      <h3>Thresholds Used</h3>
      <div class="breakdown-row"><span style="font-size:12px">Face excellent</span><span class="count" style="font-size:12px">≥ 0.75</span></div>
      <div class="breakdown-row"><span style="font-size:12px">Face acceptable</span><span class="count" style="font-size:12px">≥ 0.55</span></div>
      <div class="breakdown-row"><span style="font-size:12px">CLIP pass</span><span class="count" style="font-size:12px">≥ 0.20</span></div>
      <div class="breakdown-row"><span style="font-size:12px">Blur sharp</span><span class="count" style="font-size:12px">≥ 100</span></div>
      <div class="breakdown-row"><span style="font-size:12px">Artifact/Safety</span><span class="count" style="font-size:12px">diff &gt; −0.05</span></div>
    </div>
  </div>

  <!-- Results table -->
  <div class="section-title">Per-Row Results</div>
  <div style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Prompt</th>
        <th style="text-align:center">Reference</th>
        <th style="text-align:center">Generated</th>
        <th style="text-align:center">Model</th>
        <th style="text-align:center">Face Similarity</th>
        <th style="text-align:center">Prompt Adherence</th>
        <th style="text-align:center">Quality</th>
        <th style="text-align:center">Artifacts</th>
        <th style="text-align:center">Safety</th>
        <th style="text-align:center">Style</th>
        <th>Evaluated At</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
  </div>

</div><!-- /.body -->

<div class="footer">
  BLAST · Face: insightface/buffalo_l · Prompt: openclip/ViT-B-32 · Quality: OpenCV Laplacian · Artifacts + Safety + Style: blip-vqa-base · Report generated {generated_at}
</div>

</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from BLAST JSON results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing JSON result files (default: ./results/)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output HTML file (default: <results-dir>/report.html)")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: results directory not found: {args.results_dir}")
        raise SystemExit(1)

    output_path = args.output or (args.results_dir / "report.html")

    print(f"Reading results from: {args.results_dir.resolve()}")
    results = load_results(args.results_dir)

    if not results:
        print("No JSON result files found.")
        raise SystemExit(1)

    print(f"Loaded {len(results)} result(s).")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = build_html(results, generated_at)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Report written → {output_path.resolve()}")


if __name__ == "__main__":
    main()
