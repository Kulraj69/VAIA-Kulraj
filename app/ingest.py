import argparse
import uuid
from typing import List
import os

from .llm import embed_texts
from .vector_store import upsert_vectors


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    if max_chars <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


async def main(file_path: str, section: str, max_chars: int, overlap_chars: int) -> None:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        parts_text: List[str] = []
        for page in reader.pages:
            parts_text.append(page.extract_text() or "")
        content = "\n".join(parts_text)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

    parts = chunk_text(content, max_chars=max_chars, overlap_chars=overlap_chars)
    embeddings = await embed_texts(parts)

    to_upsert = []
    for idx, (text, vec) in enumerate(zip(parts, embeddings)):
        to_upsert.append(
            {
                "chunk_id": f"{section}_chunk_{idx}_{uuid.uuid4().hex[:8]}",
                "text": text,
                "embedding": vec.tolist(),
                "metadata": {"section": section, "source_file": file_path, "index": idx},
            }
        )
    await upsert_vectors(to_upsert)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--section", default="document")
    parser.add_argument("--max_chars", type=int, default=2000)
    parser.add_argument("--overlap_chars", type=int, default=200)
    args = parser.parse_args()

    import asyncio

    asyncio.run(main(args.file, args.section, args.max_chars, args.overlap_chars))


