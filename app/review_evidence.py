"""Shared, battle-tested normalization for literal review evidence."""

from __future__ import annotations

import re


DIACRITICS = "\u0610-\u061a\u0640\u064b-\u065f\u0670\u06d6-\u06ed"
NUMERIC_ATOM_RE = re.compile(
    r"(?<!\w)[0-9٠-٩]+"
    r"(?:\s*[-./:%٪+−]\s*[0-9٠-٩]+|[%٪])*(?!\w)"
)


def _protect_numeric_atom(match):
    """
    Keep medically meaningful number punctuation distinct.

    The ordinary evidence normalizer intentionally removes punctuation, which
    would otherwise make ``1-5`` indistinguishable from ``1.5``. Replacing a
    complete numeric atom with a deterministic word token keeps it one word
    for the 3-12 word limit while preserving its exact punctuation.
    """
    atom = re.sub(r"\s+", "", match.group(0))
    encoded = "".join(f"{ord(character):x}z" for character in atom)
    return f" numericatom{encoded} "


def normalize_evidence(text):
    """
    Normalize only safe orthographic and punctuation differences.

    This does not stem words, accept synonyms, or apply fuzzy matching.
    """
    if not isinstance(text, str):
        return ""
    value = NUMERIC_ATOM_RE.sub(_protect_numeric_atom, text)
    value = re.sub(f"[{DIACRITICS}]", "", value)
    value = value.translate(
        str.maketrans(
            {
                "أ": "ا",
                "إ": "ا",
                "آ": "ا",
                "ٱ": "ا",
                "ى": "ي",
                "ة": "ه",
            }
        )
    )
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalized_quote_in_source(quote_norm, source_norm):
    """
    Require a contiguous sequence of complete normalized words.

    The only narrow tolerance is one optional Arabic waw/fa conjunction at
    the first word boundary.
    """
    if not quote_norm or not source_norm:
        return False
    if f" {quote_norm} " in f" {source_norm} ":
        return True
    words = quote_norm.split()
    pattern = r"(?<!\S)[وف]?" + re.escape(words[0])
    for word in words[1:]:
        pattern += r"\s" + re.escape(word)
    pattern += r"(?!\S)"
    if re.search(pattern, source_norm):
        return True
    if words[0][:1] in {"و", "ف"} and len(words[0]) > 3:
        stripped = " ".join([words[0][1:]] + words[1:])
        if f" {stripped} " in f" {source_norm} ":
            return True
    return False


def validate_evidence_quote(quote, sources, min_words=3, max_words=12):
    """
    Return an empty string when a quote is valid, otherwise an Arabic reason.

    Every source is checked separately so an invalid quote cannot be assembled
    from the end of one field and the start of another.
    """
    quote_norm = normalize_evidence(quote)
    words = quote_norm.split()
    if not quote_norm:
        return "دليل الاقتباس فارغ"
    if not min_words <= len(words) <= max_words:
        return (
            f"دليل الاقتباس لازم يكون من {min_words} إلى {max_words} كلمة "
            f"بعد التطبيع، والحالي {len(words)}"
        )
    source_norms = [
        normalize_evidence(source) for source in sources if isinstance(source, str)
    ]
    if not any(
        normalized_quote_in_source(quote_norm, source_norm)
        for source_norm in source_norms
    ):
        return "دليل الاقتباس مش موجود حرفيا بعد التطبيع الآمن في المصدر المطلوب"
    return ""
