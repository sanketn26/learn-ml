"""A stand-in model that can only repeat what retrieval gave it.

A fake model with one canned reply ("Settings > API Keys ...") answers
every question with API-key instructions — including a question about a
failed payment that retrieved the password-reset runbook. The golden
tickets can't catch that if they only check doc ids.

`extractive_model()` reads the prompt's retrieved context and answers with
the top runbook's first sentence, quoting its id. It never knows anything
the context doesn't say. So if retrieval hands it the wrong runbook, the
answer is visibly wrong, and a golden ticket that checks the answer's
text fails — the way it would with a real model grounded on bad context.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

CONTEXT_LINE = re.compile(r"^\[([\w-]+)\] (.+)$", re.M)


def _answer(prompt) -> AIMessage:
    text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    hits = CONTEXT_LINE.findall(text)
    if not hits:
        return AIMessage(content="I don't know — nothing in the runbooks covers this.")
    doc_id, body = hits[0]
    first_sentence = re.split(r"(?<=[.!?])\s", body, maxsplit=1)[0]
    return AIMessage(content=f"Per the {doc_id} runbook: {first_sentence}")


def extractive_model() -> RunnableLambda:
    """Drop-in for the chat model in `prompt | model | StrOutputParser()`."""
    return RunnableLambda(_answer, name="extractive_model")
