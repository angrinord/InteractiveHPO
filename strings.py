"""
Loads user-facing strings from strings.json.

To add a new language, add a top-level object to strings.json whose key is
the locale code and whose values override whichever English strings differ.
Unchanged strings can be omitted — they will be inherited from "en".

Interpolation conventions:
- Strings with {placeholders} are Python format strings; call .format(**kwargs).
- Hover-template strings containing %{{x}}-style double-braces are also
  Python format strings: the double braces become single braces (Plotly
  syntax) after .format() is called.
- Hover-template strings with NO Python placeholders (e.g. hover_incumbent)
  contain literal Plotly %{x} syntax and must NOT be passed through .format().
"""

from __future__ import annotations

import json
from pathlib import Path

_data: dict = json.loads(
    Path(__file__).with_suffix(".json").read_text(encoding="utf-8")
)
_en: dict[str, str] = _data.get("en", {})


LOCALES: dict[str, str] = {"English": "en", "Deutsch": "de", "Español": "es"}


def get_strings(locale: str = "en") -> dict[str, str]:
    """Return strings for *locale*, falling back to English for missing keys."""
    active: dict[str, str] = _data.get(locale, {})
    return {**_en, **active}
