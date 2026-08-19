"""The tool contract. Five tools, on purpose — narrow enough that a small
fine-tuned model can be *reliable* at all of them, unlike a general model
improvising a schema from a prompt.
"""

from __future__ import annotations

TOOLS: dict[str, dict] = {
    "review_diff": {
        "description": "Review a code diff and decide approve vs request changes.",
        "parameters": {
            "diff": (str,),
            "file_path": (str,),
        },
        "required": ["diff"],
    },
    "find_potential_bugs": {
        "description": "Scan a code snippet for likely bugs (not style).",
        "parameters": {
            "code": (str,),
            "language": (str,),
        },
        "required": ["code"],
    },
    "suggest_fix": {
        "description": "Propose a concrete patch for a named issue in a snippet.",
        "parameters": {
            "code": (str,),
            "issue": (str,),
        },
        "required": ["code", "issue"],
    },
    "explain_error": {
        "description": "Diagnose a stack trace or error message: root cause + likely fix.",
        "parameters": {
            "traceback": (str,),
        },
        "required": ["traceback"],
    },
    "check_style_or_conventions": {
        "description": "Flag style/convention violations against a named ruleset.",
        "parameters": {
            "code": (str,),
            "ruleset": (str,),
        },
        "required": ["code"],
    },
}


def tool_schema() -> list[dict]:
    """The function-calling schema you'd hand to a model (FunctionGemma, an
    OpenAI/Anthropic-style API, or llama.cpp's grammar-constrained mode)."""
    schema = []
    for name, spec in TOOLS.items():
        schema.append(
            {
                "name": name,
                "description": spec["description"],
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in spec["parameters"]},
                    "required": spec["required"],
                },
            }
        )
    return schema
