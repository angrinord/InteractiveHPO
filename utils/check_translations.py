#!/usr/bin/env python3
"""
check_translations.py — verify that every string in _STRINGS has a
non-empty translation in each non-English .po file.

Usage:
    python utils/check_translations.py

Exit code 0 if all translations are present, 1 if any are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.strings import _STRINGS, LOCALES

_LOCALE_DIR = Path(__file__).parent.parent / "locale"


def _parse_po(path: Path) -> dict[str, str]:
    """Parse a .po file and return {msgid: msgstr} for translated entries."""
    catalog: dict[str, str] = {}
    msgid: str | None = None
    msgstr: str | None = None
    in_msgstr = False

    def _unquote(s: str) -> str:
        return s[1:-1].replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")

    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if line.startswith("msgid "):
            in_msgstr = False
            msgid = _unquote(line[6:].strip())
        elif line.startswith("msgstr "):
            in_msgstr = True
            msgstr = _unquote(line[7:].strip())
        elif line.startswith('"') and line.endswith('"'):
            # Continuation line
            if in_msgstr and msgstr is not None:
                msgstr += _unquote(line)
        elif not line:
            if msgid and msgstr:
                catalog[msgid] = msgstr
            msgid = msgstr = None
            in_msgstr = False

    # Handle last entry (file may not end with a blank line)
    if msgid and msgstr:
        catalog[msgid] = msgstr

    return catalog


def main() -> int:
    non_english = {label: code for label, code in LOCALES.items() if code != "en"}
    exit_code = 0

    for label, code in non_english.items():
        po_path = _LOCALE_DIR / code / "LC_MESSAGES" / "app.po"

        if not po_path.exists():
            print(f"[{label}] MISSING .po file: {po_path}")
            exit_code = 1
            continue

        catalog = _parse_po(po_path)
        missing = [
            (key, en_text)
            for key, en_text in _STRINGS.items()
            if not catalog.get(en_text)
        ]

        if missing:
            print(f"[{label}] {len(missing)} untranslated string(s):")
            for key, en_text in missing:
                print(f"  {key!r:40s}  {en_text!r}")
            exit_code = 1
        else:
            print(f"[{label}] OK — all {len(_STRINGS)} strings translated.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
