VERDICT_PROMPT_TEMPLATE = """You are an expert VAR (Video Assistant Referee) analyst. Given a description of an in-match incident and the relevant IFAB Laws of the Game text, predict the most likely VAR decision BEFORE it is officially announced.

INCIDENT DESCRIPTION:
{incident_text}

RELEVANT LAW TEXT (retrieved from IFAB Laws of the Game 2025/26):
{retrieved_law_text}

First, write a brief step-by-step ANALYSIS (3-5 sentences):
1. Quote the exact phrase(s) in the incident description that matter.
2. Quote or closely paraphrase the exact criterion/criteria in the law text above that those phrases map to.
3. State plainly whether each mapped criterion is SATISFIED or NOT SATISFIED by the incident's facts - follow the law's literal wording over general intuition about what feels "accidental" or "minor."

Then, after your analysis, output a line containing only "FINAL ANSWER:" followed by a JSON object in this exact schema, no markdown code fences:
{{
  "predicted_ruling": "<one of: Offside / No Offside / Penalty / No Penalty / Red Card / Yellow Card / No Card / Goal Disallowed / Goal Stands / VAR Review - No Clear Error>",
  "confidence_percent": <integer 0-100>,
  "law_citation": "<exact Law number and short title, e.g. 'Law 11 - Offside'>",
  "rationale": "<2-3 sentence explanation grounded in the cited law text, referencing the specific facts in the incident>",
  "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"]
}}

Be decisive: give a single primary predicted_ruling with a calibrated confidence score reflecting how clear-cut vs. borderline the incident is. Do not hedge with multiple rulings."""

JSON_RETRY_SUFFIX = (
    "\n\nYour previous response was not valid JSON. Return ONLY the JSON object "
    "described above, with no extra text, no commentary, and no markdown code fences."
)


def build_verdict_prompt(incident_text: str, retrieved_law_text: str) -> str:
    return VERDICT_PROMPT_TEMPLATE.format(
        incident_text=incident_text.strip(),
        retrieved_law_text=retrieved_law_text.strip(),
    )
