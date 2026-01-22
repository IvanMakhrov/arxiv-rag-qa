import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import requests


def save_metadata(data: list, metadata_path: str):
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with Path.open(str(metadata_path), "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {metadata_path}")


def fetch_arxiv_pdfs(
    category: str = "",
    start_date: str = "",
    target_count: int = 0,
    results_per_request: int = 0,
    raw_pdf_dir: str = "",
    metadata_path: str = "",
) -> list[dict]:
    """
    Download arXiv papers as PDFs and return metadata list.
    Does NOT parse to JSON.
    """
    raw_pdf_dir = Path(raw_pdf_dir)
    raw_pdf_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15"
    }

    downloaded = {}
    start_index = 0

    print(f"Target: {target_count} papers from '{category}' since {start_date[:8]}")

    while len(downloaded) < target_count:
        remaining = target_count - len(downloaded)
        batch_size = min(results_per_request, remaining)

        search_query = f"cat:{category} AND submittedDate:[{start_date} TO 999912312359]"
        encoded_query = quote(search_query)

        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={encoded_query}&"
            f"sortBy=submittedDate&"
            f"sortOrder=descending&"
            f"start={start_index}&"
            f"max_results={batch_size}"
        )

        print(
            f"Requesting papers {start_index}-{start_index + batch_size - 1} "
            f"(total so far: {len(downloaded)})"
        )

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"API request failed: {e}. Retrying in 5s")
            time.sleep(5)
            continue

        root = ET.fromstring(response.content)
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")

        if not entries:
            print("No more papers available")
            break

        for entry in entries:
            try:
                paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
                title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
                summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
                authors = [
                    author.find("{http://www.w3.org/2005/Atom}name").text
                    for author in entry.findall("{http://www.w3.org/2005/Atom}author")
                ]
                pdf_path = raw_pdf_dir / f"{paper_id}.pdf"

                if paper_id not in downloaded:
                    if not pdf_path.exists():
                        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                        pdf_resp = requests.get(pdf_url, headers=headers, timeout=10)
                        pdf_resp.raise_for_status()
                        with pdf_path.open("wb") as f:
                            f.write(pdf_resp.content)
                        print(f"Downloaded: {paper_id}")
                    else:
                        print(f"Already exists: {paper_id}")

                    downloaded[paper_id] = {
                        "arxiv_id": paper_id,
                        "title": title,
                        "abstract": summary,
                        "authors": authors,
                        "pdf_path": str(pdf_path),
                    }
                    time.sleep(1)
            except Exception as e:
                print(f"Error processing {paper_id}: {e}")
                continue

        start_index += batch_size
        if len(entries) < batch_size:
            break

        metadata_list = [downloaded[pid] for pid in downloaded]
        save_metadata(metadata_list, metadata_path)

    return len(list(downloaded.values()))
