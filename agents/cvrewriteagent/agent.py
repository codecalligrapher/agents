"""
Gradio frontend for CVRewriteAgent.
Usage: python app.py
Requires: pip install gradio requests
Assumes: ollama serve is running on localhost:11434
"""

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
import gradio as gr

from pathlib import Path

# ── Ollama config ───────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:30b"


# ── Agent (self-contained) ──────────────────────────────────────────────────


@dataclass
class CVRewriteAgent:
    model: str = MODEL

    def _complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 16384,
            },
        }
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(OLLAMA_URL, json=payload, timeout=3000)
        resp.raise_for_status()
        data = resp.json()

        # ollama returns {"error": "..."} on model-not-found etc., with 200 status
        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error']}")

        msg = data.get("message", {})
        content = (msg.get("content") or "").strip()

        # qwen3 uses a thinking field — if content is empty, the model
        # may have dumped everything into thinking (rare, but happens
        # with format=json on some quantizations)
        if not content:
            thinking = (msg.get("thinking") or "").strip()
            if thinking:
                content = thinking
            else:
                raise RuntimeError(
                    f"Ollama returned empty content. Full response: "
                    f"{json.dumps(data, indent=2)[:500]}"
                )

        return content

    @staticmethod
    def _parse_json(raw: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if m:
                return json.loads(m.group(0))
            raise ValueError(f"No valid JSON in model output: {raw[:200]}")

    @staticmethod
    def split_latex(tex_source: str) -> Tuple[str, str]:
        m = re.search(r"(\\begin\{document\})", tex_source)
        if not m:
            raise ValueError(r"No \begin{document} found in LaTeX source")
        preamble = tex_source[: m.end()]
        body = tex_source[m.end() :]
        body = re.sub(r"\\end\{document\}\s*$", "", body).strip()
        return preamble, body

    def analyze(self, body_tex: str, job: Dict[str, str]) -> Dict[str, Any]:
        system = (
            "You are an ATS optimization specialist. You output ONLY valid JSON, "
            "no markdown, no commentary."
        )
        user = f"""Analyze this CV against the job description.
The CV is in LaTeX — read through the markup to understand the actual content.

Return JSON matching this schema:
{{
  "verdict": "Strong Fit" | "Good Fit" | "Needs Repositioning" | "Poor Fit",
  "confidence": 0.0-1.0,
  "gaps": [
    {{
      "section": "experience|skills|summary|projects|education",
      "issue": "what's wrong or missing",
      "fix": "specific actionable change",
      "priority": "high|medium|low"
    }}
  ],
  "missing_keywords": ["keyword1", "keyword2"],
  "narrative_angle": "one sentence: the story this CV should tell for this role"
}}

JOB TITLE: {job.get('title_raw', '')}
JOB DESCRIPTION:
{job.get('description', '')[:4000]}

CV BODY (LaTeX):
{body_tex[:6000]}"""

        data = self._parse_json(
            self._complete(
                system, user, temperature=0.2, max_tokens=32768, json_mode=True
            )
        )
        for key in ("verdict", "confidence", "gaps", "missing_keywords"):
            if key not in data:
                raise KeyError(f"Model output missing required key: '{key}'")
        data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
        return data

    def rewrite(
        self, body_tex: str, job: Dict[str, str], analysis: Dict[str, Any]
    ) -> str:
        system = r"""You are a professional resume writer who outputs LaTeX.

RULES:
- Output ONLY the rewritten LaTeX body content. No \documentclass, no \begin{document}, no \end{document}.
- Preserve every custom command and macro call from the original.
- Preserve the sectioning structure.
- You may reorder items WITHIN a section, rephrase bullet text, and add missing keywords where factually truthful.
- Do NOT invent experience, metrics, or skills.
- Do NOT wrap output in markdown code fences."""

        gaps_block = json.dumps(analysis["gaps"], indent=2)
        user = f"""Rewrite this CV body for the target role.

EDITING PLAN:
- Narrative angle: {analysis.get('narrative_angle', 'N/A')}
- Missing keywords to weave in where truthful: {', '.join(analysis.get('missing_keywords', []))}
- Gaps:
{gaps_block}

TARGET ROLE: {job.get('title_raw', '')}
JOB DESCRIPTION (excerpt):
{job.get('description', '')[:3000]}

ORIGINAL CV BODY (LaTeX):
{body_tex}

Output the rewritten LaTeX body. Nothing else."""

        raw = self._complete(system, user, temperature=0.4, max_tokens=4096)
        return re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", raw.strip())

    def run(self, tex_source: str, job: Dict[str, str]) -> Dict[str, Any]:
        preamble, body = self.split_latex(tex_source)
        analysis = self.analyze(body, job)
        rewritten_body = self.rewrite(body, job, analysis)
        full_tex = f"{preamble}\n{rewritten_body}\n\\end{{document}}"
        return {"analysis": analysis, "rewritten_tex": full_tex}


agent = CVRewriteAgent()


# ── Gradio callback ────────────────────────────────────────────────────────


def process(tex_file, job_title: str, job_description: str):
    """Main callback wired to the Gradio UI."""
    if tex_file is None:
        raise gr.Error("Upload a .tex file first.")
    if not job_description.strip():
        raise gr.Error("Paste a job description.")

    tex_source = Path(tex_file).read_text(encoding="utf-8", errors="replace")
    job = {"title_raw": job_title.strip(), "description": job_description.strip()}

    result = agent.run(tex_source, job)

    # format analysis for display
    analysis = result["analysis"]
    verdict_line = (
        f"**{analysis['verdict']}** (confidence: {analysis['confidence']:.0%})"
    )
    narrative = analysis.get("narrative_angle", "")
    keywords = ", ".join(analysis.get("missing_keywords", [])) or "None"

    gaps_md = ""
    for g in analysis.get("gaps", []):
        gaps_md += (
            f"- **[{g['priority'].upper()}]** `{g['section']}` — {g['issue']}\n"
            f"  → {g['fix']}\n"
        )

    analysis_md = (
        f"## {verdict_line}\n\n"
        f"**Narrative angle:** {narrative}\n\n"
        f"**Missing keywords:** {keywords}\n\n"
        f"### Gaps\n{gaps_md or 'None identified.'}"
    )

    # write rewritten .tex to a temp file for download
    out_path = Path(tempfile.mkdtemp()) / "rewritten_cv.tex"
    out_path.write_text(result["rewritten_tex"], encoding="utf-8")

    return analysis_md, result["rewritten_tex"], str(out_path)


# ── UI ──────────────────────────────────────────────────────────────────────

with gr.Blocks(title="CV Rewriter") as app:
    gr.Markdown(
        "# CV ↔ JD Rewriter\nUpload your `.tex` resume and paste a job description."
    )

    with gr.Row():
        with gr.Column():
            tex_input = gr.File(label="CV (.tex)", file_types=[".tex"])
            title_input = gr.Textbox(
                label="Job Title", placeholder="Senior Backend Engineer"
            )
            jd_input = gr.Textbox(
                label="Job Description",
                lines=12,
                placeholder="Paste the full job description here…",
            )
            run_btn = gr.Button("Analyze & Rewrite", variant="primary")

        with gr.Column():
            analysis_output = gr.Markdown(label="Analysis")
            tex_output = gr.Code(label="Rewritten LaTeX", language="latex")
            download = gr.File(label="Download .tex")

    run_btn.click(
        fn=process,
        inputs=[tex_input, title_input, jd_input],
        outputs=[analysis_output, tex_output, download],
    )

if __name__ == "__main__":
    app.launch()
