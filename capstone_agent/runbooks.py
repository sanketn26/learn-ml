"""CloudWave runbooks and a keyword-overlap retriever (LangChain Week 4).

There is deliberately no runbook for large exports: CW-1847 is an open
incident, and the honest answer to "why does my 150k-row export time
out?" is "I don't know yet — a human has it."
"""

from __future__ import annotations

RUNBOOKS = {
    "api-keys": "API keys live in Settings > API Keys. Generate a key, copy it once, send it as "
                "Authorization: Bearer. Rotate keys every 90 days.",
    "password-reset": "Forgot password: click Forgot Password, enter your email, open the reset link. "
                      "New passwords need 12+ characters. Five failed logins lock the account for an hour.",
    "plans": "Free: 100 requests/day. Pro: $99/month, 100K requests/day. Enterprise: custom pricing. "
             "Cancel anytime. Invoices live in Settings > Billing.",
}

MIN_SCORE = 0.25  # below this overlap, a hit is noise: say "I don't know"


def tokens(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2}


def retrieve(question: str, k: int = 2) -> list[tuple[float, str, str]]:
    """[(score, doc_id, text)], best first; hits under MIN_SCORE are dropped."""
    q = tokens(question)
    scored = [(len(q & tokens(text)) / max(len(q), 1), doc_id, text) for doc_id, text in RUNBOOKS.items()]
    scored.sort(key=lambda hit: hit[0], reverse=True)
    return [hit for hit in scored[:k] if hit[0] >= MIN_SCORE]
