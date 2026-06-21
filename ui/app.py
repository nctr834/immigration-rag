"""One-page Gradio UI over the RAG API.

A thin HTTP client: it POSTs the question to the API's /query and shows the
answer with its sources. Point it at a running API with API_URL (defaults to a
local instance).

    API_URL=http://localhost:8000 python ui/app.py
"""

from __future__ import annotations

import os

import gradio as gr
import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 60.0  # cold starts and generation can be slow on free tiers

EXAMPLES = [
    "Can the Form I-864 Affidavit of Support requirement be waived?",
    "After entering on a K-1 visa, who must the beneficiary marry to adjust status?",
    "What income level must a sponsor demonstrate on Form I-864?",
    "Does the medical exam from the K-1 consular process carry over to the I-485?",
]


def ask(question: str) -> tuple[str, str]:
    """POST the question to the API and return (answer, sources-as-markdown)."""
    question = (question or "").strip()
    if not question:
        return "Enter a question.", ""
    try:
        resp = httpx.post(
            f"{API_URL}/query", json={"question": question}, timeout=TIMEOUT
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", e.response.text)
        return f"API error ({e.response.status_code}): {detail}", ""
    except httpx.HTTPError as e:
        return f"Could not reach the API at {API_URL}: {e}", ""

    data = resp.json()
    sources = data.get("sources", [])
    sources_md = "\n".join(f"- {s}" for s in sources) if sources else "_none_"
    return data.get("answer", ""), sources_md


with gr.Blocks(title="Immigration RAG") as demo:
    gr.Markdown(
        "# Immigration RAG\n"
        "Ask about USCIS immigration forms. Answers are grounded in the "
        "instruction documents, with sources."
    )
    question = gr.Textbox(label="Question", placeholder="Can the I-864 be waived?")
    ask_btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=6)
    sources = gr.Markdown(label="Sources")
    gr.Examples(EXAMPLES, inputs=question)

    ask_btn.click(ask, inputs=question, outputs=[answer, sources])
    question.submit(ask, inputs=question, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("UI_PORT", "7860")))
