"""One-time pipeline: parse the IFAB Laws of the Game PDF with Docling into
structured Markdown + chunked JSON suitable for retrieval.

Run: python -m src.ingest_laws
"""
import json
import re

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.config import RAW_PDF_PATH, LAW_SECTIONS_MD_PATH, LAW_SECTIONS_JSON_PATH

# The IFAB document's heading format is inconsistent across the PDF: some
# chapter starts are "## Law 12" (number only, title on the next heading),
# others are just the canonical title with no "Law N" prefix at all (e.g.
# "## Offside"), and "Law N. Title" cross-references appear standalone inside
# unrelated appendix sections. We detect chapter boundaries by matching either
# a "Law N" heading or one of the 17 canonical titles below (case-insensitive,
# normalized), and treat every other heading as a subsection that stays under
# the current law rather than resetting it - this avoids fragmenting a law's
# text every time an unrelated cross-reference heading appears.
CANONICAL_LAW_TITLES: dict[int, str] = {
    1: "the field of play",
    2: "the ball",
    3: "the players",
    4: "the players' equipment",
    5: "the referee",
    6: "the other match officials",
    7: "the duration of the match",
    8: "the start and restart of play",
    9: "the ball in and out of play",
    10: "determining the outcome of a match",
    11: "offside",
    12: "fouls and misconduct",
    13: "free kicks",
    14: "the penalty kick",
    15: "the throw-in",
    16: "the goal kick",
    17: "the corner kick",
}
_TITLE_TO_LAW_NO = {v: k for k, v in CANONICAL_LAW_TITLES.items()}
# Some titles render with word order scrambled by Docling's column detection
# (e.g. "Match Officials The Other" instead of "The Other Match Officials").
_TITLE_TO_LAW_NO["match officials the other"] = 6

LAW_NUMBER_RE = re.compile(r"^#{1,3}\s*Law\s+(\d{1,2})\.?\s*[-–:]?\s*(.*)$", re.IGNORECASE)
HEADING_RE = re.compile(r"^#{1,3}\s*(.+)$")
MAX_CHUNK_CHARS = 2500


def _match_law_boundary(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    number_match = LAW_NUMBER_RE.match(stripped)
    if number_match:
        law_no = int(number_match.group(1))
        title = number_match.group(2).strip() or CANONICAL_LAW_TITLES.get(law_no, "")
        return law_no, title

    heading_match = HEADING_RE.match(stripped)
    if heading_match:
        normalized = heading_match.group(1).strip().lower()
        if normalized in _TITLE_TO_LAW_NO:
            law_no = _TITLE_TO_LAW_NO[normalized]
            return law_no, CANONICAL_LAW_TITLES[law_no]
    return None


def convert_pdf_to_markdown() -> str:
    # The IFAB Laws of the Game PDF is digitally typeset (not scanned), so OCR
    # is unnecessary - disabling it also sidesteps a broken default OCR engine
    # config in this docling version.
    pdf_options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )
    result = converter.convert(str(RAW_PDF_PATH))
    markdown = result.document.export_to_markdown()
    LAW_SECTIONS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAW_SECTIONS_MD_PATH.write_text(markdown, encoding="utf-8")
    return markdown


def split_into_law_sections(markdown: str) -> list[dict]:
    lines = markdown.splitlines()
    sections: list[dict] = []
    current_law_no = None
    current_title = None
    current_lines: list[str] = []

    def flush():
        if current_law_no is None or not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            return
        for subsection_text in _split_on_subheadings(text):
            for chunk_text in _chunk_text(subsection_text, MAX_CHUNK_CHARS):
                sections.append(
                    {
                        "law_no": current_law_no,
                        "title": current_title,
                        "text": chunk_text,
                    }
                )

    for line in lines:
        boundary = _match_law_boundary(line)
        if boundary:
            law_no, title = boundary
            if law_no != current_law_no:
                flush()
                current_law_no = law_no
                current_title = title
                current_lines = []
                continue
        current_lines.append(line)
    flush()
    return sections


_SUBHEADING_RE = re.compile(r"^#{1,3}\s+\S")


def _split_on_subheadings(text: str) -> list[str]:
    """Split a law's full text on its internal "## ..." subheadings (e.g.
    "## Handling the ball") so each topic becomes its own retrievable chunk
    instead of being diluted inside a long multi-topic chunk - critical for
    embedding precision on a small corpus.
    """
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if _SUBHEADING_RE.match(line.strip()) and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    chunks, buf = [], ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 > max_chars and buf:
            chunks.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# Docling's layout model crashes (std::bad_alloc) on the PDF pages covering
# Laws 13-17 in this environment, regardless of OCR/table-structure settings -
# an environment-specific rendering issue on those pages, not a parsing logic
# bug. Per the project's documented fallback for Docling parsing gaps, these
# few short law sections are hand-curated from the IFAB Laws of the Game
# 2025/26 (summarized, not verbatim) so retrieval still has correct grounding
# text for penalty-kick scenarios even though Docling couldn't reach them.
MANUAL_LAW_SUPPLEMENT: list[dict] = [
    {
        "law_no": 13,
        "title": "Free Kicks",
        "text": (
            "All free kicks are either direct or indirect. For a direct free kick, "
            "a goal may be scored directly against the offending team. For an "
            "indirect free kick, a goal can only be scored if the ball subsequently "
            "touches another player (of either team) before entering the goal. At "
            "the moment the kick is taken, opponents must be at least 9.15m (10 yds) "
            "from the ball until it is in play, unless they are on their own goal line "
            "between the goalposts."
        ),
    },
    {
        "law_no": 14,
        "title": "The Penalty Kick",
        "text": (
            "A penalty kick is awarded when a direct free kick offence is committed "
            "by a player inside their own penalty area, or when they commit an "
            "offence inside their own penalty area for which an indirect free kick "
            "would otherwise be awarded against an opponent who, at that moment, is "
            "inside the penalty area. The ball is placed on the penalty mark; the "
            "defending goalkeeper must remain on the goal line, facing the kicker, "
            "between the goalposts until the ball is kicked. All other players must "
            "be outside the penalty area, outside the penalty arc, and behind the "
            "penalty mark until the kick is taken. If a foul that denies an opponent "
            "an obvious goal-scoring opportunity (DOGSO) occurs inside the penalty "
            "area and the offence is an attempt to play the ball, a penalty is "
            "awarded and the offending player is shown a red card; if the offence "
            "involves no attempt to play the ball (e.g. holding, pulling, or an "
            "off-the-ball act), it is also a sending-off plus a penalty."
        ),
    },
    {
        "law_no": 15,
        "title": "The Throw-in",
        "text": (
            "A throw-in is awarded when the whole ball passes over a touchline, "
            "either on the ground or in the air, and is awarded to the opponents of "
            "the player who last touched the ball. The thrower must face the field "
            "of play, have part of each foot on or behind the touchline or on the "
            "ground outside it, and use both hands to deliver the ball from behind "
            "and over the head from the point where it left the field of play. A "
            "goal cannot be scored directly from a throw-in."
        ),
    },
    {
        "law_no": 16,
        "title": "The Goal Kick",
        "text": (
            "A goal kick is awarded when the whole ball, having last touched an "
            "attacking player, passes over the goal line (excluding the goal), "
            "either on the ground or in the air. The ball is kicked from any point "
            "within the goal area and is in play once it is kicked and clearly "
            "moves; opponents must remain outside the penalty area until the ball "
            "is in play. A goal may be scored directly from a goal kick against "
            "either team."
        ),
    },
    {
        "law_no": 17,
        "title": "The Corner Kick",
        "text": (
            "A corner kick is awarded when the whole ball, having last touched a "
            "defending player, passes over the goal line (excluding the goal), "
            "either on the ground or in the air. The ball is placed inside the "
            "corner arc nearest to where it crossed the line; opponents must stay "
            "at least 9.15m (10 yds) from the ball until it is in play. A goal may "
            "be scored directly from a corner kick."
        ),
    },
]


def merge_manual_supplement(sections: list[dict]) -> list[dict]:
    covered_law_nos = {s["law_no"] for s in sections}
    for entry in MANUAL_LAW_SUPPLEMENT:
        if entry["law_no"] not in covered_law_nos:
            sections.append(dict(entry))
    return sections


def main():
    print(f"Converting {RAW_PDF_PATH} with Docling (this can take a minute)...")
    markdown = convert_pdf_to_markdown()
    print(f"Wrote markdown to {LAW_SECTIONS_MD_PATH} ({len(markdown)} chars)")

    sections = split_into_law_sections(markdown)
    if not sections:
        raise RuntimeError(
            "No law headings found in the converted markdown. Inspect "
            "law_sections.md and adjust _match_law_boundary, or fall back "
            "to a fully hand-curated law_sections.json (see README limitations)."
        )

    sections = merge_manual_supplement(sections)
    sections.sort(key=lambda s: s["law_no"])

    for i, section in enumerate(sections):
        section["section_id"] = f"law{section['law_no']:02d}_{i:03d}"

    LAW_SECTIONS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAW_SECTIONS_JSON_PATH.write_text(json.dumps(sections, indent=2), encoding="utf-8")
    print(f"Wrote {len(sections)} law chunks to {LAW_SECTIONS_JSON_PATH}")


if __name__ == "__main__":
    main()
