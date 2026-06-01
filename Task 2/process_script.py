"""
Replaces Diablo universe terms in raw knowledge base files using terms_map.json,
and saves the processed files to knowledge_base/processed/.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "knowledge_base" / "raw"
PROCESSED_DIR = ROOT / "knowledge_base" / "processed"
TERMS_MAP_PATH = ROOT / "terms_map.json"


def load_terms(path: Path) -> dict[str, str]:
    """Flatten the nested terms_map into a single {original: replacement} dict."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    terms: dict[str, str] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            terms.update(value)
        elif isinstance(value, str):
            terms[key] = value

    return terms


def build_replacer(terms: dict[str, str]):
    """
    Build a single-pass regex replacer.
    Longer terms are matched first to avoid partial replacements
    (e.g. 'Black Soulstone' before 'Soulstone').
    Matches are word-boundary-aware for whole-word terms.
    """
    sorted_terms = sorted(terms.keys(), key=len, reverse=True)

    pattern = "|".join(re.escape(t) for t in sorted_terms)
    regex = re.compile(pattern)

    def replace(match: re.Match) -> str:
        return terms[match.group(0)]

    return regex, replace


def process_file(src: Path, dst: Path, regex: re.Pattern, replace_fn) -> int:
    text = src.read_text(encoding="utf-8")
    result, count = regex.subn(replace_fn, text)
    dst.write_text(result, encoding="utf-8")
    return count


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    terms = load_terms(TERMS_MAP_PATH)
    print(f"Loaded {len(terms)} term replacements.")

    regex, replace_fn = build_replacer(terms)

    files = list(RAW_DIR.glob("*.md"))
    if not files:
        print(f"No .md files found in {RAW_DIR}")
        return

    total_replacements = 0
    for src in sorted(files):
        dst = PROCESSED_DIR / src.name
        count = process_file(src, dst, regex, replace_fn)
        total_replacements += count
        print(f"  {src.name}: {count} replacements -> {dst}")

    print(f"\nDone. {len(files)} files processed, {total_replacements} replacements total.")


if __name__ == "__main__":
    main()
