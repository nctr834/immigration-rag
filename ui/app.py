"""Gradio UI for the RAG pipeline, mounted onto the FastAPI app at /ui.

It calls generate() in-process rather than POSTing to the API's own /query: the
UI and the API run in the same process, so a round-trip through HTTP would just
serialize JSON to talk to itself. See api.py for the mount.
"""

from __future__ import annotations

import os
import sys

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate import generate

EXAMPLES = [
    "Can the Form I-864 Affidavit of Support requirement be waived?",
    "After entering on a K-1 visa, who must the beneficiary marry to adjust status?",
    "What income level must a sponsor demonstrate on Form I-864?",
    "Does the medical exam from the K-1 consular process carry over to the I-485?",
]


def ask(question: str) -> tuple[str, str]:
    """Answer the question and return (answer, sources+disclaimer-as-markdown)."""
    question = (question or "").strip()
    if not question:
        return "Enter a question.", ""
    try:
        result = generate(question)
    except Exception as e:  # surface the error in the UI rather than 500-ing
        return f"Error: {e}", ""
    if result.sources:
        sources_md = "\n".join(
            f'- **{c.source}** — "{c.quote}"' for c in result.sources
        )
    else:
        sources_md = "_none_"
    sources_md += f"\n\n_{result.disclaimer}_"
    return result.answer, sources_md


def build_demo() -> gr.Blocks:
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
    return demo


demo = build_demo()


if __name__ == "__main__":
    # Standalone run (UI only, no REST endpoint). Normally the UI is served by
    # api.py at /ui; this is just for working on the UI in isolation.
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("UI_PORT", "7860")))
