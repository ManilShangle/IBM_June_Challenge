# Fix Backlog

Generated from /simplify, /code-review, /security-review audits.

---

## /simplify

- [x] **S1** `app/app.py · render_verdict` — `hdr_bg` if/elif/else duplicates `ruling_color` logic. Extracted into `ruling_tint()` backed by `_RULING_TINTS` dict keyed on color constants.
- [x] **S2** `app/app.py · render_verdict` — `plain_english_law` conditional buried inside HTML f-string. Moved to a `plain_html` variable computed before the card string.
- [x] **S3** `app/app.py · run_prediction_from_video` — `visual_desc = ""` in except block was dead (already ""). Removed; early `return` makes failure explicit.
- [x] **S4** `app/app.py` — `run_prediction_from_video` and `run_prediction_from_text` duplicated the full pipeline. Extracted shared `_run_prediction(situation, visual_desc)`.
- [x] **S5** `app/app.py · run_prediction_from_video` — `visual_desc if visual_desc else None` → `visual_desc or None`.
- [x] **S6** `app/app.py · Famous Incidents loop` — redundant outer `<div>` wrapper around iframe removed. Single `components.html()` call with inline border-radius.
- [x] **S7** `app/app.py` — YouTube ID extraction replaced with `safe_yt_id()` using `_YT_ID_FROM_URL` regex that handles `youtube.com/watch`, `youtu.be/`, and `/shorts/`. Returns `None` on failure; UI shows a warning instead of a broken embed.
- [x] **S8** `src/predictor.py · VerdictResult` — `confidence_percent` and `key_factors` dead fields removed. `__post_init__` None-guard removed with them. Skipped smoke test updated to drop `confidence_percent` assertion.

---

## /code-review

- [x] **C1** Same as S5 — fixed.
- [x] **C2** Same as S7 — fixed.
- [x] **C3** `confidence_percent` field removed entirely (S8). No silent 0 any more.
- [x] **C4** `run_prediction_from_video` except block now does an explicit `return` after `st.warning`, making failure unambiguous to callers and preventing a silent empty-description call to the predictor.

---

## /security-review

- [x] **V1 HIGH XSS** `render_verdict` — all LLM-generated fields (`predicted_ruling`, `law_citation`, `rationale`, `plain_english_law`) wrapped with `html.escape()` before interpolation into `unsafe_allow_html=True` HTML.
- [x] **V2 MED-HIGH XSS** `safe_yt_id()` validates with `_YT_ID_RE = r'^[A-Za-z0-9_-]{11}$'`. Any URL that doesn't yield a conforming 11-char ID returns `None` and renders a warning, not an injected attribute.
