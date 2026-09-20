from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Callable


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk])
            chunks.append(chunk.strip())

        return chunks

class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self._split(text.strip(), self.separators)

    def _split_text_with_separator(self, text: str, separator: str) -> list[str]:
        """Tách chuỗi bằng separator nhưng bảo toàn ký tự/dấu câu."""
        if separator == "":
            return list(text)

        if separator == ". ":
            # Tránh làm mất dấu chấm câu: chỉ tách tại khoảng trắng sau dấu chấm
            parts = re.split(r"(?<=\.) ", text)
            return [p for p in parts if p]

        # Đối với \n\n, \n, " ": split thông thường
        parts = text.split(separator)
        return [p for p in parts if p]

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Gom các mảnh nhỏ lại thành các chunk không vượt quá chunk_size."""
        docs: list[str] = []
        current_doc: list[str] = []
        total_len = 0

        # Xác định chuỗi nối khi gom
        join_sep = "" if separator == ". " else separator

        for piece in splits:
            piece_len = len(piece)
            sep_len = len(join_sep) if current_doc else 0

            if total_len + sep_len + piece_len <= self.chunk_size:
                current_doc.append(piece)
                total_len += sep_len + piece_len
            else:
                if current_doc:
                    docs.append(join_sep.join(current_doc).strip())
                current_doc = [piece]
                total_len = piece_len

        if current_doc:
            docs.append(join_sep.join(current_doc).strip())

        return [doc for doc in docs if doc]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        current_text = current_text.strip()
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []

        # Nếu không còn separator nào, bắt buộc cắt cứng theo ký tự
        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        # Tìm separator đầu tiên có mặt trong văn bản
        chosen_sep = None
        next_separators_idx = len(remaining_separators)

        for i, sep in enumerate(remaining_separators):
            if sep == "" or sep in current_text:
                chosen_sep = sep
                next_separators_idx = i + 1
                break

        # Nếu không có separator nào khớp (hiếm khi xảy ra vì luôn có ""), fallback cắt cứng
        if chosen_sep is None:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        # Tách chuỗi theo separator đã chọn
        splits = self._split_text_with_separator(current_text, chosen_sep)
        next_separators = remaining_separators[next_separators_idx:]

        # Đệ quy cho những phần vẫn còn vượt quá chunk_size
        good_splits: list[str] = []
        for piece in splits:
            if len(piece) <= self.chunk_size:
                good_splits.append(piece)
            else:
                # Mẩu này vẫn quá lớn, tiếp tục dùng separator cấp thấp hơn
                further_splits = self._split(piece, next_separators)
                good_splits.extend(further_splits)

        # Gom các mảnh nhỏ lại với nhau
        return self._merge_splits(good_splits, chosen_sep)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (norm_a * norm_b)




class HeaderSectionChunker:
    """
    Split structured markdown and policy documents by headers (#, ##, ###)
    and numbered section/article headings (e.g. '1. ', 'Bước 1:').
    
    Preserves policy articles and legal clauses as atomic units. If a section
    exceeds max_chunk_size, it gracefully sub-chunks using RecursiveChunker.
    """

    def __init__(self, max_chunk_size: int = 600) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Match markdown headers (#, ##) or numbered sections (e.g., '1. ', 'Bước 1:')
        pattern = r"(?:\n\s*\n|\n)(?=(?:#{1,4}\s+|\d+\.\s+[A-ZÀ-ỸĐ]|Bước\s+\d+:|[IVXLCDM]+\.\s+))"
        raw_sections = [s.strip() for s in re.split(pattern, text) if s.strip()]

        chunks: list[str] = []
        fallback = RecursiveChunker(chunk_size=self.max_chunk_size)
        for sec in raw_sections:
            if len(sec) <= self.max_chunk_size:
                chunks.append(sec)
            else:
                chunks.extend(fallback.chunk(sec))
        return chunks


class SemanticChunker:
    """
    Split text into semantically cohesive chunks by detecting drops in
    cosine similarity between consecutive sentences.
    
    Supports Gemini embeddings (using GEMINI_API_KEY from environment) or a
    custom embedding function, falling back gracefully to feature hashing if offline.
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], list[float]] | None = None,
        similarity_threshold: float = 0.65,
        max_chunk_size: int = 800,
        min_sentences_per_chunk: int = 1,
        use_gemini: bool = False,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self._embedding_fn = embedding_fn
        self.use_gemini = use_gemini
        self._cache: dict[str, list[float]] = {}

    def _hash_embed(self, text: str) -> list[float]:
        words = re.findall(r"[\w]+", text.lower())
        if not words:
            return [0.0] * 64
        vec = [0.0] * 64
        tokens = words + [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        for w in tokens:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % 64
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _embed_sentences(self, sentences: list[str]) -> list[list[float]]:
        if not sentences:
            return []

        if self._embedding_fn is not None:
            return [self._embedding_fn(s) for s in sentences]

        if self.use_gemini:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
                    
                    results: list[list[float]] = []
                    # Process in batches of 40 to stay well under payload and rate limits
                    batch_size = 40
                    for b_start in range(0, len(sentences), batch_size):
                        batch = sentences[b_start : b_start + batch_size]
                        res = client.models.embed_content(model=model_name, contents=batch)
                        for emb in res.embeddings:
                            results.append([float(v) for v in emb.values])
                    return results
                except Exception as err:
                    print(f"[WARN] Gemini batch embed failed ({err}), falling back to hash embeddings.")

        return [self._hash_embed(s) for s in sentences]

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            return sentences if sentences else []

        embeddings = self._embed_sentences(sentences)

        chunks: list[str] = []
        current_chunk_sentences: list[str] = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(len(sentences) - 1):
            next_sent = sentences[i + 1]
            sim = compute_similarity(embeddings[i], embeddings[i + 1])

            is_semantic_boundary = sim < self.similarity_threshold
            exceeds_size = current_len + len(next_sent) + 1 > self.max_chunk_size

            if (is_semantic_boundary or exceeds_size) and len(current_chunk_sentences) >= self.min_sentences_per_chunk:
                chunks.append(" ".join(current_chunk_sentences).strip())
                current_chunk_sentences = [next_sent]
                current_len = len(next_sent)
            else:
                current_chunk_sentences.append(next_sent)
                current_len += len(next_sent) + 1

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences).strip())

        return chunks



class ChunkingStrategyComparator:
    """Run built-in and extended chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200, include_advanced: bool = False) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }
        if include_advanced:
            strategies["header_section"] = HeaderSectionChunker(max_chunk_size=chunk_size).chunk(text)
            strategies["semantic"] = SemanticChunker(max_chunk_size=chunk_size).chunk(text)

        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result

