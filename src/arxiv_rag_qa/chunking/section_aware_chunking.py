import hashlib
import json
import re
from typing import Any

import boto3
import fitz

from utils.setup_logger import setup_logger

logger = setup_logger(__name__)

CANONICAL_SECTIONS = ["Abstract", "Introduction", "Main text", "Conclusion", "References"]

SECTION_PATTERNS = {
    "Abstract": [
        r"^abstract\s*[:.]?\s*$",
        r"^\s*abstract\s*$",
    ],
    "Introduction": [
        r"^1\s+introduction\s*$",
        r"^introduction\s*[:.]?\s*$",
        r"^\s*introduction\s*$",
    ],
    "Conclusion": [
        r"^\d*\s*conclusion[s]?\s*[:.]?\s*$",
        r"^\s*conclusion\s*$",
        r"^\d*\s*future\s+work\s*$",
    ],
    "References": [
        r"^references\s*[:.]?\s*$",
        r"^\s*references\s*$",
        r"^bibliography\s*$",
        r"^works\s+cited\s*$",
    ],
}

MIN_HEADING_LENGTH = 5
MAX_HEADING_LENGTH = 150

MIN_SECTION_SIZE = 200

MAX_HEADING_LINE_LENGTH = 50

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
]


def get_minio_client() -> boto3.client:
    """Создание клиента S3/MinIO."""
    return boto3.client("s3")


def extract_full_text_with_pymupdf(pdf_bytes: bytes) -> str:
    """Извлечение полного текста из PDF-документа в памяти."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text("text")
    doc.close()
    return full_text


def classify_header(header_text: str) -> str | None:
    """Определение типа секции по заголовку."""
    header_lower = header_text.lower().strip()

    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, header_lower, re.IGNORECASE):
                return section_name

    if any(kw in header_lower for kw in MAIN_TEXT_KEYWORDS):
        return "Main text"

    if re.match(r"^\d+(\.\d+)*\s+", header_text.strip()):
        return "Main text"

    return None


def find_target_heading_position_enhanced(lines: list[str]) -> int:
    """
    Enhanced target heading position detection that looks for:
    1. "2. Related Work" on consecutive lines (highest priority)
    2. "2. Related Work" on the same line
    3. Any numbered heading starting with "2." followed by significant text
    4. Just "2" on its own line
    """
    introduction_start = -1
    for idx, line in enumerate(lines):
        stripped = line.strip().lower()
        if "introduction" in stripped or (
            stripped and stripped[0].isdigit() and "introduction" in stripped
        ):
            introduction_start = idx
            break

    if introduction_start == -1:
        return -1

    for idx in range(introduction_start + 1, len(lines) - 1):
        line1 = lines[idx].strip()
        line2 = lines[idx + 1].strip()

        if line1 == "2." and "Related Work" in line2:
            return idx

        if "2." in line1 and "Related Work" in line1:
            return idx

        if re.match(r"^2\.\s*\w", line1):
            return idx

        if line1 == "2" and line2 and len(line2) > MIN_HEADING_LENGTH:
            return idx

    return -1


def extract_sections_dynamic(text: str) -> list[dict[str, Any]]:  # noqa: C901, PLR0912
    """
    Extract sections using dynamic detection of the first numbered heading after Introduction.
    FIXED: Ensures mutually exclusive sections with proper boundary ordering.
    """
    lines = text.split("\n")

    marker_positions = {"abstract": [], "introduction": [], "conclusion": [], "references": []}

    for idx, line in enumerate(lines):
        stripped = line.strip().lower()

        if len(stripped) <= MAX_HEADING_LINE_LENGTH and not any(
            p in stripped for p in [".", "?", "!", ":"]
        ):
            if re.match(r"^abstract\s*[:.]?\s*$", stripped) or stripped == "abstract":
                marker_positions["abstract"].append(idx)
            elif (
                re.match(r"^(\d+\s+)?introduction\s*[:.]?\s*$", stripped)
                or stripped == "introduction"
            ):
                marker_positions["introduction"].append(idx)
            elif (
                re.match(r"^(\d+\s+)?conclusion[s]?\s*[:.]?\s*$", stripped)
                or stripped == "conclusion"
                or "future work" in stripped
            ):
                marker_positions["conclusion"].append(idx)
            elif re.match(r"^references\s*[:.]?\s*$", stripped) or stripped in [
                "references",
                "bibliography",
                "works cited",
            ]:
                marker_positions["references"].append(idx)

    abstract_start = marker_positions["abstract"][0] if marker_positions["abstract"] else -1
    introduction_start = (
        marker_positions["introduction"][0] if marker_positions["introduction"] else -1
    )
    conclusion_start = marker_positions["conclusion"][0] if marker_positions["conclusion"] else -1
    references_start = marker_positions["references"][0] if marker_positions["references"] else -1

    main_text_start = find_target_heading_position_enhanced(lines)

    if main_text_start == -1:
        return extract_sections_simple(text)

    boundaries = []

    if abstract_start > 0:
        boundaries.append((0, abstract_start - 1, "Header"))

    if abstract_start >= 0:
        end = introduction_start - 1 if introduction_start > abstract_start else len(lines) - 1
        if end >= abstract_start:
            boundaries.append((abstract_start, end, "Abstract"))

    if introduction_start >= 0 and introduction_start >= (max(abstract_start, 0)):
        end = main_text_start - 1 if main_text_start > introduction_start else len(lines) - 1
        if end >= introduction_start:
            boundaries.append((introduction_start, end, "Introduction"))

    if main_text_start >= 0:
        next_boundary = len(lines)
        if conclusion_start > main_text_start:
            next_boundary = min(next_boundary, conclusion_start)
        if references_start > main_text_start:
            next_boundary = min(next_boundary, references_start)

        end = next_boundary - 1
        if end >= main_text_start:
            boundaries.append((main_text_start, end, "Main text"))

    if conclusion_start >= 0 and conclusion_start >= main_text_start:
        end = references_start - 1 if references_start > conclusion_start else len(lines) - 1
        if end >= conclusion_start:
            boundaries.append((conclusion_start, end, "Conclusion"))

    if (
        references_start >= 0
        and references_start >= (conclusion_start if conclusion_start >= 0 else main_text_start)
        and references_start <= len(lines) - 1
    ):
        boundaries.append((references_start, len(lines) - 1, "References"))

    if not boundaries:
        return extract_sections_simple(text)

    sections = []
    for start, end, section_name in boundaries:
        start_line = max(0, min(start, len(lines) - 1))
        end_line = max(start_line, min(end, len(lines) - 1))

        section_lines = lines[start_line : end_line + 1]
        section_text = "\n".join(section_lines).strip()

        if section_text:
            sections.append(
                {
                    "section": section_name,
                    "text": section_text,
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )

    return sections


def extract_sections_simple(text: str) -> list[dict[str, Any]]:  # noqa: C901
    """Extract sections by clear boundaries - ensures non-overlapping sections."""
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
    """
    Базовая предобработка текста:
    - Удаление переносов слов через дефис
    - Нормализация пробелов
    - Удаление лишних цифр-нумераторов в начале строк
    - Сохранение важных переносов строк для разделения секций
    """
    text = re.sub(r"-\n\s*", "", text)

    text = re.sub(r"\n[1-9]\s+", "\n", text)

    lines = text.split("\n")
    normalized_lines = []
    for line in lines:
        normalized_line = re.sub(r"\s+", " ", line.strip())
        if normalized_line:
            normalized_lines.append(normalized_line)

    return "\n".join(normalized_lines)


def split_text_recursive(  # noqa: C901, PLR0912
    text: str, chunk_size: int = 512, chunk_overlap: int = 50, separators: list[str] | None = None
) -> list[str]:
    """Рекурсивное разбиение текста на чанки."""
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

                if len(current_chunk + part_with_sep) <= chunk_size * 1.1:
                    current_chunk += part_with_sep
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.rstrip())

                    if len(part_with_sep) <= chunk_size * 1.1:
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
                if len(chunk) > chunk_size and len(chunk) > chunk_size * 1.2:
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


def _generate_chunk_id(arxiv_id: str, section: str, chunk_idx: int, text: str) -> str:
    """Генерация уникального ID для чанка."""
    hash_input = f"{arxiv_id}:{section}:{chunk_idx}:{text[:100]}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    section_clean = section.lower().replace(" ", "_")
    return f"{arxiv_id}_{section_clean}_{chunk_idx:04d}_{hash_value}"


def chunking(  # noqa: C901, PLR0912
    bucket_name: str,
    pdf_dir: str,
    chunk_dir: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
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
            sections = extract_sections_dynamic(cleaned_text)

            logger.info(f"Found {len(sections)} sections: {[s['section'] for s in sections]}")

            merged_sections = []
            current_section = None

            for section in sections:
                section_text = section["text"]
                section_size = len(section_text)
                section_type = section["section"]

                if (
                    section_size < MIN_SECTION_SIZE
                    and current_section is not None
                    and current_section["section"] == section_type
                ):
                    current_section["text"] += "\n\n" + section_text
                    current_section["end_line"] = section["end_line"]
                else:
                    if current_section is not None:
                        merged_sections.append(current_section)
                    current_section = section.copy()

            if current_section is not None:
                merged_sections.append(current_section)

            logger.info(f"Merged into {len(merged_sections)} sections for better chunking")

            chunk_idx = 0
            for section in merged_sections:
                section_name = section["section"]
                section_text = section["text"]

                if not section_text.strip():
                    continue

                section_chunks = split_text_recursive(
                    section_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )

                for chunk_text in section_chunks:
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
