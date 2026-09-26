"""CloudWave runbooks and a keyword-overlap retriever (LangChain Week 4).

There is deliberately no runbook for large exports: CW-1847 is an open
incident, and the honest answer to "why does my 150k-row export time
out?" is "I don't know yet — a human has it."

A hit needs at least MIN_SHARED content words in common with the question.
One shared word is noise: "How do I fix a failed payment?" shares only
"failed" with the password-reset runbook ("Five failed logins lock the
account"), and answering it from that runbook is confidently wrong.
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

STOPWORDS = {"the", "and", "how", "can", "you", "does", "what", "why", "for", "are", "with", "this", "that", "your",
             "our", "get", "keep", "keeps", "any", "not"}
MIN_SHARED = 2  # content words a question and a runbook must share before the runbook may answer it


def tokens(text: str) -> set[str]:
    """Lowercased content words, plural 's' stripped so "keys" meets "key"."""
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words if len(w) > 2 and w not in STOPWORDS}


def retrieve(question: str, k: int = 2) -> list[tuple[float, str, str]]:
    """[(score, doc_id, text)], best first; runbooks sharing fewer than MIN_SHARED words are dropped."""
    q = tokens(question)
    scored = []
    for doc_id, text in RUNBOOKS.items():
        shared = q & tokens(text)
        if len(shared) >= MIN_SHARED:
            scored.append((len(shared) / max(len(q), 1), doc_id, text))
    scored.sort(key=lambda hit: hit[0], reverse=True)
    return scored[:k]
