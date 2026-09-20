"""Benchmark all chunking strategies (built-in and advanced) on real ecommerce docs.

Usage:
    .venv/bin/python scripts/compare_chunking.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import ChunkingStrategyComparator

DEFAULT_DOCS = [
    "data/ecommerce/chinh-sach-bao-mat.md",
    "data/ecommerce/chinh-sach-tra-hang-hoan-tien.md",
    "data/ecommerce/dieu-khoan-dich-vu.md",
]


def main() -> int:
    include_adv = "--basic" not in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = args or DEFAULT_DOCS
    comparator = ChunkingStrategyComparator()

    print("=" * 68)
    print("SO SÁNH CÁC CHIẾN LƯỢC CHUNKING TRÊN TÀI LIỆU E-COMMERCE (SHOPEE)")
    print(f"Bao gồm chiến lược nâng cao (Header/Section, Semantic): {include_adv}")
    print("=" * 68)

    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        text = path.read_text(encoding="utf-8")
        # Strip frontmatter if present for fair text measurement
        body = text.split("---", 2)[2].strip() if text.startswith("---") and len(text.split("---", 2)) >= 3 else text

        result = comparator.compare(body, chunk_size=500, include_advanced=include_adv)

        print(f"\n=== {path.name} (Văn bản gốc: {len(body):,} ký tự) ===")
        print(f"{'strategy':<16} {'count':>6} {'avg_len':>9} {'min':>6} {'max':>6}")
        print("-" * 47)
        for name, stats in result.items():
            chunks = stats["chunks"]
            lengths = [len(c) for c in chunks] or [0]
            print(
                f"{name:<16} {stats['count']:>6} {stats['avg_length']:>9.1f}"
                f" {min(lengths):>6} {max(lengths):>6}"
            )
    print("\n" + "=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
