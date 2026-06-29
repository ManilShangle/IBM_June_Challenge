VERDICT_PROMPT_TEMPLATE = """You are a VAR analyst. Predict the ruling for this football incident using the IFAB Laws provided.

INCIDENT:
{incident_text}

IFAB LAWS:
{retrieved_law_text}

Respond with ONLY this JSON object, no other text, no markdown:
{{"predicted_ruling": "...", "law_citation": "...", "plain_english_law": "...", "rationale": "..."}}

Rules:
- predicted_ruling must be exactly one of: Offside / No Offside / Penalty / No Penalty / Red Card / Yellow Card / No Card / Goal Disallowed / Goal Stands / VAR Review - No Clear Error
- law_citation: e.g. "Law 11 - Offside"
- plain_english_law: one sentence explaining the law in simple terms a non-expert understands
- rationale: one or two sentences saying why this ruling applies to this specific incident"""

JSON_RETRY_SUFFIX = (
    "\n\nReturn ONLY the JSON object above. No other text."
)


def build_verdict_prompt(incident_text: str, retrieved_law_text: str) -> str:
    return VERDICT_PROMPT_TEMPLATE.format(
        incident_text=incident_text.strip(),
        retrieved_law_text=retrieved_law_text.strip(),
    )
