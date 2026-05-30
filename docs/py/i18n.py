"""Tiny bilingual helper shared by the backend modules.

All user-facing strings are written inline in both Traditional Chinese (zh,
the canonical/default) and English (en). `t(lang, zh, en)` picks one. Keeping
both literals side by side means the branching/threshold logic stays in one
place and only the wording differs by language.
"""
from __future__ import annotations


def t(lang, zh, en):
    """Return the English string when lang == 'en', otherwise the Chinese one."""
    return en if str(lang).lower().startswith("en") else zh


def norm(lang) -> str:
    """Normalise an arbitrary lang value to 'en' or 'zh'."""
    return "en" if str(lang).lower().startswith("en") else "zh"
