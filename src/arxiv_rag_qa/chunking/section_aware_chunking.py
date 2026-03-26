import hashlib
import json
import re
from typing import Any

import boto3
import fitz

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)

SECTION_PATTERNS = {
    "Abstract": [
        r"^abstract\s*[:.]?\s*$",
        r"^\s*abstract\s*$",
        r"^abstract\s*[-–—]\s*",  # noqa: RUF001
        r"^abstract\s+$",
    ],
    "Introduction": [
        r"^1\s+introduction\s*$",
        r"^introduction\s*[:.]?\s*$",
        r"^\s*introduction\s*$",
        r"^1\.?\s+introduction\s*$",
        r"^i\.?\s+introduction\s*$",
        r"^i\.?\s+[\w\s]+$",
    ],
    "Related Work": [
        r"^2\s+related\s+work\s*$",
        r"^related\s+work\s*[:.]?\s*$",
        r"^\s*related\s+work\s*$",
        r"^2\.?\s+related\s+work\s*$",
        r"^previous\s+work\s*[:.]?\s*$",
        r"^\s*previous\s+work\s*$",
        r"^literature\s+review\s*[:.]?\s*$",
    ],
    "Background": [
        r"^background\s*[:.]?\s*$",
        r"^\s*background\s*$",
    ],
    "Methodology": [
        r"^3\s+method(ology)?\s*$",
        r"^method(ology)?\s*[:.]?\s*$",
        r"^\s*method(ology)?\s*$",
        r"^3\.?\s+method(ology)?\s*$",
        r"^approach\s*[:.]?\s*$",
        r"^\s*approach\s*$",
        r"^proposed\s+method\s*[:.]?\s*$",
        r"^research\s+method(ology)?\s*$",
    ],
    "Methods": [
        r"^3\s+methods?\s*$",
        r"^methods?\s*[:.]?\s*$",
        r"^\s*methods?\s*$",
    ],
    "Model": [
        r"^model\s*[:.]?\s*$",
        r"^\s*model\s*$",
        r"^our\s+model\s*[:.]?\s*$",
        r"^architecture\s*[:.]?\s*$",
        r"^\s*architecture\s*$",
        r"^framework\s*[:.]?\s*$",
        r"^\s*frameworks?\s*$",
    ],
    "Data": [
        r"^4\s+data\s*$",
        r"^data\s*[:.]?\s*$",
        r"^\s*data\s*$",
        r"^dataset(s)?\s*[:.]?\s*$",
        r"^\s*dataset(s)?\s*$",
        r"^data\s+collection\s*[:.]?\s*$",
        r"^experimental\s+setup\s*[:.]?\s*$",
        r"^setup\s*[:.]?\s*$",
    ],
    "Experiments": [
        r"^5\s+experiments?\s*$",
        r"^experiments?\s*[:.]?\s*$",
        r"^\s*experiments?\s*$",
        r"^evaluation\s*[:.]?\s*$",
        r"^\s*evaluation\s*$",
        r"^experimental\s+results\s*[:.]?\s*$",
    ],
    "Results": [
        r"^results\s*[:.]?\s*$",
        r"^\s*results\s*$",
        r"^findings\s*[:.]?\s*$",
        r"^\s*findings\s*$",
        r"^empirical\s+results\s*[:.]?\s*$",
        r"^outcomes\s*[:.]?\s*$",
    ],
    "Discussion": [
        r"^6\s+discussion\s*$",
        r"^discussion\s*[:.]?\s*$",
        r"^\s*discussion\s*$",
        r"^analysis\s*[:.]?\s*$",
        r"^\s*analysis\s*$",
        r"^results\s+and\s+discussion\s*[:.]?\s*$",
    ],
    "Conclusion": [
        r"^\d*\s*conclusion[s]?\s*[:.]?\s*$",
        r"^\s*conclusion\s*$",
        r"^\d*\s*future\s+work\s*$",
        r"^\d*\s*summary\s*[:.]?\s*$",
        r"^\s*summary\s*$",
    ],
    "References": [
        r"^references\s*[:.]?\s*$",
        r"^\s*references\s*$",
        r"^bibliography\s*$",
        r"^works\s+cited\s*$",
        r"^reference\s+list\s*$",
        r"^literature\s+cited\s*$",
    ],
    "Appendix": [
        r"^appendix\s*[:.]?\s*$",
        r"^\s*appendix\s*$",
        r"^supplementary\s+(material|info)\s*$",
        r"^appendices\s*$",
    ],
    "Acknowledgments": [
        r"^acknowledgments?\s*[:.]?\s*$",
        r"^\s*acknowledgments?\s*$",
        r"^acknowledgements?\s*[:.]?\s*$",
        r"^\s*acknowledgements?\s*$",
    ],
}

MAIN_TEXT_KEYWORDS = [
    "methods",
    "methodology",
    "approach",
    "model",
    "data",
    "dataset",
    "experiments",
    "results",
    "analysis",
    "discussion",
    "evaluation",
    "related work",
    "preliminaries",
    "framework",
    "system",
    "architecture",
    "background",
    "motivation",
    "problem",
    "task",
    "training",
    "inference",
    "implementation",
    "procedure",
    "study",
    "design",
    "participants",
    "measurement",
    "study design",
]

MIN_HEADING_LENGTH = 2
MAX_HEADING_LENGTH = 200
MAX_HEADING_LINE_LENGTH = 100
MAX_HEADING_WORDS = 10
MIN_SECTION_SIZE = 200
MAX_SHORT_WORD_LENGTH = 3

SECTION_PRIORITY = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Background",
    "Methodology",
    "Methods",
    "Model",
    "Data",
    "Experiments",
    "Results",
    "Discussion",
    "Analysis",
    "Conclusion",
    "Summary",
    "References",
    "Appendix",
    "Acknowledgments",
]

ALLOWED_LOWERCASE_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "for",
    "nor",
    "on",
    "at",
    "to",
    "from",
    "by",
    "in",
    "of",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "can",
    "could",
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
}

EXCLUSION_PATTERNS = [
    r"^figure\s*\d+",
    r"^fig\.?\s*\d+",
    r"^table\s*\d+",
    r"^tab\.?\s*\d+",
    r"^algorithm\s*\d+",
    r"^algo\.?\s*\d+",
    r"^equation\s*\d+",
    r"^eq\.?\s*\d+",
    r"^listing\s*\d+",
    r"^photo\s*\d+",
    r"^diagram\s*\d+",
    r"^algorithm\s*\d+",
    r"^procedure\s*\d+",
]

PDF_ARTIFACTS = [
    r"(?i)\n\s*arxiv:\s*\d+\.\d+(?:v\d+)?\s*\n",
    r"(?i)\n\s*date:\s*.+\n",
    r"(?i)\n\s*doi:\s*.+\n",
    r"^\s*-\s*\d+\s*-\s*$",
    r"^\s*\d+\s*$",
    r"(?i)^\s*page\s+\d+\s*$",
    r"(?i)^\s*page\s+\d+\s+of\s+\d+\s*$",
    r"^\s*\d+\s+of\s+\d+\s*$",
]

CONFIDENCE_HIGH = 1.0
CONFIDENCE_MEDIUM_HIGH = 0.9
CONFIDENCE_MEDIUM = 0.8
CONFIDENCE_LOW = 0.6
CONFIDENCE_FILTER_THRESHOLD = 0.4

CONFIDENCE_ADJUSTMENT_HIGH = 0.9
CONFIDENCE_ADJUSTMENT_MEDIUM = 0.7

MAX_HEADING_CHECK_LENGTH = 50

CHUNK_SIZE_DEFAULT = 512
CHUNK_OVERLAP_DEFAULT = 50
CHUNK_SIZE_FACTOR = 1.1
CHUNK_SIZE_FACTOR_RECURSIVE = 1.2


def get_minio_client():
    return boto3.client("s3")


def extract_full_text_with_pymupdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text("text")
    doc.close()
    return full_text


def is_excluded_line(line: str) -> bool:
    line_lower = line.lower().strip()
    return any(re.match(pattern, line_lower) for pattern in EXCLUSION_PATTERNS)


def is_heading_line(  # noqa: C901, PLR0911, PLR0912
    line: str, prev_line: str | None = None, next_line: str | None = None
) -> bool:
    if not line or len(line) > MAX_HEADING_LINE_LENGTH:
        return False

    stripped = line.strip()

    if stripped.endswith("!") or stripped.endswith("?") or stripped.endswith(":"):
        return False

    if is_excluded_line(stripped):
        return False

    if re.match(
        r"^(\d+[\.\)]?|[A-Za-z][\.\)]?|[IVXLCDMivxlcdm]+[\.\)]?|"
        r"\([ivxlcdm]+\))\s+[A-Za-z0-9]",
        stripped,
    ):
        return True

    if stripped.isupper():
        return True

    words = stripped.split()
    if 1 <= len(words) <= MAX_HEADING_WORDS:
        valid = True
        for word in words:
            if word.isupper():
                continue
            if (
                word.islower()
                and len(word) <= MAX_SHORT_WORD_LENGTH
                and word in ALLOWED_LOWERCASE_WORDS
            ):
                continue
            if "-" in word and len(word) > 1:
                parts = word.split("-")
                for p in parts:
                    if not p:
                        continue
                    if p.isupper():
                        continue
                    if (
                        p.islower()
                        and len(p) <= MAX_SHORT_WORD_LENGTH
                        and p.lower() in ALLOWED_LOWERCASE_WORDS
                    ):
                        continue
                    if not p[0].isupper():
                        valid = False
                        break
                if not valid:
                    break
                continue
            if word and not word[0].isupper():
                valid = False
                break
        if valid:
            return True

    if prev_line is not None and next_line is not None:
        prev_stripped = prev_line.strip()
        if is_heading_line(prev_stripped) and len(stripped) < MAX_HEADING_CHECK_LENGTH:
            return True

    return False


def normalize_section_name(section_name: str) -> str:
    name_lower = section_name.lower().strip()

    mappings = {
        "background": "Background",
        "methodology": "Methodology",
        "methods": "Methods",
        "approach": "Methodology",
        "model": "Model",
        "architecture": "Model",
        "framework": "Model",
        "data": "Data",
        "dataset": "Data",
        "experiments": "Experiments",
        "evaluation": "Experiments",
        "results": "Results",
        "findings": "Results",
        "discussion": "Discussion",
        "analysis": "Analysis",
        "conclusion": "Conclusion",
        "summary": "Conclusion",
        "future work": "Conclusion",
        "references": "References",
        "bibliography": "References",
        "appendix": "Appendix",
        "acknowledgments": "Acknowledgments",
        "acknowledgements": "Acknowledgments",
        "supplementary": "Appendix",
    }

    for key, canonical in mappings.items():
        if key in name_lower:
            return canonical

    return section_name


def classify_header(  # noqa: C901, PLR0911, PLR0912
    header_text: str, context_lines: list[str] | None = None
) -> str | None:
    header_lower = header_text.lower().strip()
    header_clean = header_text.strip()

    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, header_lower, re.IGNORECASE):
                return section_name

    numbered_match = re.match(r"^(\d+(\.\d+)*)\s+", header_clean)
    if numbered_match:
        level = len(numbered_match.group(1).split("."))
        if level > 1:
            return None
        heading_text = header_clean[numbered_match.end() :].strip().lower()

        keyword_to_section = {
            "related work": "Related Work",
            "background": "Background",
            "method": "Methodology",
            "methods": "Methods",
            "approach": "Methodology",
            "model": "Model",
            "data": "Data",
            "dataset": "Data",
            "experiment": "Experiments",
            "evaluation": "Experiments",
            "result": "Results",
            "results": "Results",
            "findings": "Results",
            "discussion": "Discussion",
            "analysis": "Analysis",
            "conclusion": "Conclusion",
            "summary": "Conclusion",
        }
        for kw, section in keyword_to_section.items():
            if kw in heading_text:
                return section
        return "Main text"

    for keyword in MAIN_TEXT_KEYWORDS:
        if keyword in header_lower:
            normalized = normalize_section_name(header_lower)
            if normalized in SECTION_PRIORITY:
                return normalized
            return "Main text"

    if context_lines:
        context_text = " ".join(context_lines).lower()
        for keyword in MAIN_TEXT_KEYWORDS:
            if keyword in context_text:
                normalized = normalize_section_name(keyword)
                if normalized in SECTION_PRIORITY:
                    return normalized

    return None


def extract_sections_dynamic(  # noqa: C901, PLR0912, PLR0915
    text: str, min_section_content_size: int = 200
) -> list[dict[str, Any]]:
    lines = text.split("\n")
    n = len(lines)

    detected_headings = []

    i = 0
    while i < n:
        stripped = lines[i].strip()

        if (
            not stripped
            or len(stripped) < MIN_HEADING_LENGTH
            or len(stripped) > MAX_HEADING_LINE_LENGTH
        ):
            i += 1
            continue

        if not is_heading_line(stripped):
            i += 1
            continue

        if is_excluded_line(stripped):
            i += 1
            continue

        classification = classify_header(stripped)
        if classification:
            detected_headings.append((i, i, classification, CONFIDENCE_HIGH, stripped))
            i += 1
            continue

        if i + 1 < n:
            next_line = lines[i + 1].strip()
            if next_line and len(next_line) < MAX_HEADING_LINE_LENGTH:
                combined = stripped + " " + next_line
                combined_classification = classify_header(combined)
                if combined_classification:
                    detected_headings.append(
                        (i, i + 1, combined_classification, CONFIDENCE_MEDIUM_HIGH, combined)
                    )
                    i += 2
                    continue

        numbered_match = re.match(r"^(\d+(\.\d+)*)\s+(.+)$", stripped)
        if numbered_match:
            heading_text = numbered_match.group(3).lower()
            level = len(numbered_match.group(1).split("."))
            if level == 1:
                for keyword in MAIN_TEXT_KEYWORDS:
                    if keyword in heading_text:
                        detected_headings.append((i, i, "Main text", CONFIDENCE_MEDIUM))
                        break
                else:
                    detected_headings.append((i, i, "Main text", CONFIDENCE_LOW))
                    break

        i += 1

    if not detected_headings:
        logger.warning("No headings detected, falling back to simple extraction")
        return extract_sections_simple(text)

    refined_headings = []
    for start_idx, end_idx, section_name, confidence, raw_text in detected_headings:
        if confidence < CONFIDENCE_FILTER_THRESHOLD:
            continue

        if section_name in SECTION_PRIORITY or section_name == "Main text":
            refined_headings.append((start_idx, end_idx, section_name, confidence))
        else:
            line_lower = raw_text.lower()
            for canon_section, patterns in SECTION_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, line_lower, re.IGNORECASE):
                        refined_headings.append(
                            (
                                start_idx,
                                end_idx,
                                canon_section,
                                confidence * CONFIDENCE_ADJUSTMENT_HIGH,
                            )
                        )
                        break
                else:
                    for keyword in MAIN_TEXT_KEYWORDS:
                        if keyword in line_lower:
                            refined_headings.append(
                                (
                                    start_idx,
                                    end_idx,
                                    "Main text",
                                    confidence * CONFIDENCE_ADJUSTMENT_MEDIUM,
                                )
                            )
                            break

    refined_headings.sort(key=lambda x: x[0])

    filtered_headings = []
    last_end = -1
    for start_idx, end_idx, section_name, confidence in refined_headings:
        if start_idx <= last_end:
            if filtered_headings:
                prev_start, _prev_end, _prev_section, prev_conf = filtered_headings[-1]
                if confidence > prev_conf or (confidence == prev_conf and start_idx > prev_start):
                    filtered_headings[-1] = (start_idx, end_idx, section_name, confidence)
                    last_end = end_idx
            continue
        filtered_headings.append((start_idx, end_idx, section_name, confidence))
        last_end = end_idx

    sections = []
    for i, (start_idx, end_idx, section_name, _) in enumerate(filtered_headings):  # noqa: B007
        if i + 1 < len(filtered_headings):
            next_start = filtered_headings[i + 1][0]
            section_end = next_start - 1
        else:
            section_end = n - 1

        actual_start = start_idx

        if section_end < actual_start:
            continue

        section_text = "\n".join(lines[actual_start : section_end + 1]).strip()
        if not section_text:
            continue

        sections.append(
            {
                "section": section_name,
                "text": section_text,
                "start_line": actual_start,
                "end_line": section_end,
            }
        )

    if not sections:
        return []

    merged = []
    current = sections[0].copy()
    for next_section in sections[1:]:
        if (
            next_section["section"] == current["section"]
            and next_section["start_line"] == current["end_line"] + 1
        ):
            current["text"] += "\n\n" + next_section["text"]
            current["end_line"] = next_section["end_line"]
        else:
            merged.append(current)
            current = next_section.copy()
    merged.append(current)
    sections = merged

    filtered_sections = []
    i = 0
    while i < len(sections):
        section = sections[i]
        section_content = section["text"].strip()

        if len(section_content) < min_section_content_size:
            lines_list = section_content.split("\n")
            max_short_lines = 2
            if len(lines_list) <= max_short_lines and not any(
                c in section_content.lower() for c in [". ", "! ", "? "]
            ):
                if i + 1 < len(sections):
                    sections[i + 1]["text"] = section["text"] + "\n\n" + sections[i + 1]["text"]
                    sections[i + 1]["start_line"] = section["start_line"]
                    i += 1
                    continue
                if filtered_sections:
                    filtered_sections[-1]["text"] += "\n\n" + section["text"]
                    filtered_sections[-1]["end_line"] = section["end_line"]
                    i += 1
                    continue

        filtered_sections.append(section)
        i += 1

    sections = filtered_sections

    logger.info(f"Extracted {len(sections)} sections: {[s['section'] for s in sections]}")
    return sections


def extract_sections_simple(text: str) -> list[dict[str, Any]]:  # noqa: C901
    sections = []
    lines = text.split("\n")

    markers = {}
    for idx, line in enumerate(lines):
        stripped = line.strip().lower()
        if len(stripped) <= MAX_HEADING_LINE_LENGTH and not any(
            p in stripped for p in [".", "?", "!"]
        ):
            if "abstract" in stripped and "abstract" not in markers:
                markers["abstract"] = idx
            elif (
                "introduction" in stripped
                or (stripped and stripped[0].isdigit() and "introduction" in stripped)
            ) and "introduction" not in markers:
                markers["introduction"] = idx
            elif (
                "conclusion" in stripped or "future work" in stripped
            ) and "conclusion" not in markers:
                markers["conclusion"] = idx
            elif (
                "references" in stripped or "bibliography" in stripped
            ) and "references" not in markers:
                markers["references"] = idx

    order = ["abstract", "introduction", "conclusion", "references"]
    valid_markers = [(markers[m], m) for m in order if m in markers]
    valid_markers.sort(key=lambda x: x[0])

    sections_data = []
    prev_end = -1

    for start_idx, marker_name in valid_markers:
        if start_idx > prev_end:
            sections_data.append(
                (
                    marker_name.capitalize() if marker_name != "references" else "References",
                    start_idx,
                )
            )
        prev_end = start_idx

    for i, (section_name, start_line) in enumerate(sections_data):
        end_line = sections_data[i + 1][1] - 1 if i + 1 < len(sections_data) else len(lines) - 1

        if start_line <= end_line:
            section_text = "\n".join(lines[start_line : end_line + 1]).strip()
            if section_text:
                sections.append(
                    {
                        "section": section_name,
                        "text": section_text,
                        "start_line": start_line,
                        "end_line": end_line,
                    }
                )

    if not sections and text.strip():
        sections.append(
            {
                "section": "Main text",
                "text": text.strip(),
                "start_line": 0,
                "end_line": len(lines) - 1,
            }
        )

    return sections


def preprocess_text(text: str) -> str:
    text = re.sub(r"-\n\s*", "", text)

    for pattern in PDF_ARTIFACTS:
        text = re.sub(pattern, "\n", text, flags=re.MULTILINE)

    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned = re.sub(r"\s+", " ", line.strip())
        if cleaned:
            cleaned_lines.append(cleaned)
        else:
            cleaned_lines.append("")

    return "\n".join(cleaned_lines)


def split_text_recursive(  # noqa: C901, PLR0912
    text: str,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    chunk_overlap: int = CHUNK_OVERLAP_DEFAULT,
    separators: list[str] | None = None,
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current_chunk = ""

            for part in parts:
                part_with_sep = part + (sep if sep != " " else "")

                if len(current_chunk + part_with_sep) <= chunk_size * CHUNK_SIZE_FACTOR:
                    current_chunk += part_with_sep
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.rstrip())

                    if len(part_with_sep) <= chunk_size * CHUNK_SIZE_FACTOR:
                        current_chunk = part_with_sep
                    else:
                        current_chunk = ""
                        next_separators = (
                            separators[separators.index(sep) + 1 :]
                            if sep in separators[:-1]
                            else [" "]
                        )
                        sub_chunks = split_text_recursive(
                            part, chunk_size, chunk_overlap, next_separators
                        )
                        chunks.extend(sub_chunks)

            if current_chunk.strip():
                chunks.append(current_chunk.rstrip())

            final_chunks = []
            for chunk in chunks:
                if (
                    len(chunk) > chunk_size
                    and len(chunk) > chunk_size * CHUNK_SIZE_FACTOR_RECURSIVE
                ):
                    final_chunks.extend(
                        split_text_recursive(chunk, chunk_size, chunk_overlap, separators)
                    )
                elif chunk.strip():
                    final_chunks.append(chunk)

            return final_chunks

    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def merge_chunks_by_size(  # noqa: C901, PLR0912
    chunks: list[str], min_chunk_size: int, max_chunk_size: int
) -> list[str]:
    if not chunks:
        return []

    merged_chunks = []
    buffer = ""

    for chunk_text in chunks:
        chunk_text = chunk_text.strip()  # noqa: PLW2901
        if not chunk_text:
            continue

        chunk_len = len(chunk_text)

        if chunk_len < min_chunk_size:
            if buffer:
                buffer += "\n\n" + chunk_text
            else:
                buffer = chunk_text

            if len(buffer) >= min_chunk_size:
                merged_chunks.append(buffer)
                buffer = ""
        else:
            if buffer:
                combined_len = len(buffer) + chunk_len + 2
                if combined_len <= max_chunk_size:
                    chunk_text = buffer + "\n\n" + chunk_text  # noqa: PLW2901
                    buffer = ""
                else:
                    if len(buffer) >= min_chunk_size:
                        merged_chunks.append(buffer)
                    else:
                        merged_chunks.append(buffer)
                    buffer = ""
            merged_chunks.append(chunk_text)

    if buffer:
        if merged_chunks:
            last_chunk = merged_chunks[-1]
            if len(last_chunk) + len(buffer) + 2 <= max_chunk_size:
                merged_chunks[-1] += "\n\n" + buffer
            else:
                merged_chunks.append(buffer)
        else:
            merged_chunks.append(buffer)

    return merged_chunks


def _generate_chunk_id(arxiv_id: str, section: str, chunk_idx: int, text: str) -> str:
    hash_input = f"{arxiv_id}:{section}:{chunk_idx}:{text[:100]}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    section_clean = section.lower().replace(" ", "_")
    return f"{arxiv_id}_{section_clean}_{chunk_idx:04d}_{hash_value}"


def chunking(  # noqa: C901
    bucket_name: str,
    pdf_dir: str,
    chunk_dir: str,
    chunk_size: int = 768,
    chunk_overlap: int = 100,
) -> int:
    s3_client = get_minio_client()

    paginator = s3_client.get_paginator("list_objects_v2")
    pdf_files = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=pdf_dir):
        if "Contents" in page:
            pdf_files.extend(
                [
                    obj["Key"]
                    for obj in page["Contents"]
                    if obj["Key"].endswith(".pdf") and not obj["Key"].endswith("/")
                ]
            )

    if not pdf_files:
        logger.warning(f"No PDF files found in s3://{bucket_name}/{pdf_dir}")
        return 0

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    all_chunks: list[dict[str, Any]] = []

    for pdf_key in pdf_files:
        try:
            arxiv_id = pdf_key.rstrip(".pdf").split("/")[-1]
            arxiv_id_clean = re.sub(r"v\d+$", "", arxiv_id)

            logger.info(f"Processing: {pdf_key}")

            pdf_obj = s3_client.get_object(Bucket=bucket_name, Key=pdf_key)
            pdf_bytes = pdf_obj["Body"].read()

            full_text = extract_full_text_with_pymupdf(pdf_bytes)
            if not full_text.strip():
                logger.warning(f"Empty text extracted from {pdf_key}")
                continue

            cleaned_text = preprocess_text(full_text)
            min_section_content_size = chunk_size // 2
            sections = extract_sections_dynamic(cleaned_text, min_section_content_size)

            logger.info(f"Found {len(sections)} sections: {[s['section'] for s in sections]}")

            chunk_idx = 0
            for section in sections:
                section_name = section["section"]
                section_text = section["text"]

                if not section_text.strip():
                    continue

                section_chunks = split_text_recursive(
                    section_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )

                min_chunk_size = chunk_size // 2
                max_chunk_size = chunk_size
                merged_chunks = merge_chunks_by_size(section_chunks, min_chunk_size, max_chunk_size)

                for chunk_text in merged_chunks:
                    if not chunk_text.strip():
                        continue

                    chunk = {
                        "id": _generate_chunk_id(
                            arxiv_id_clean, section_name, chunk_idx, chunk_text
                        ),
                        "text": chunk_text.strip(),
                        "metadata": {
                            "arxiv_id": arxiv_id_clean,
                            "source": pdf_key,
                            "section": section_name,
                            "chunk_idx": chunk_idx,
                            "chunk_size": len(chunk_text.strip()),
                        },
                    }
                    all_chunks.append(chunk)
                    chunk_idx += 1

            logger.info(f"  → {chunk_idx} chunks from {arxiv_id_clean} ({len(sections)} sections)")

        except Exception as e:
            logger.error(f"Failed to process {pdf_key}: {type(e).__name__}: {e}", exc_info=True)
            continue

    if all_chunks:
        jsonl_lines = [json.dumps(chunk, ensure_ascii=False) for chunk in all_chunks]
        jsonl_content = "\n".join(jsonl_lines)

        s3_client.put_object(
            Bucket=bucket_name,
            Key=chunk_dir,
            Body=jsonl_content.encode("utf-8"),
            ContentType="application/x-ndjson",
        )

        logger.info(f"Saved {len(all_chunks)} chunks to s3://{bucket_name}/{chunk_dir}")
        return len(all_chunks)

    logger.warning("No chunks were generated")
    return 0
