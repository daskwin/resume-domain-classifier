from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_WS_RE = re.compile(r"\s+")


def clean_text(
    text: str,
    *,
    lowercase: bool,
    strip: bool,
    remove_urls: bool,
    remove_emails: bool,
    remove_numbers: bool,
    keep_only_basic_punct: bool,
    max_length: int,
) -> str:
    value = text

    value = value.replace("\\n", " ").replace("\\t", " ").replace("\n", " ").replace("\t", " ")

    if strip:
        value = value.strip()
    if lowercase:
        value = value.lower()
    if remove_urls:
        value = _URL_RE.sub(" ", value)
    if remove_emails:
        value = _EMAIL_RE.sub(" ", value)
    if remove_numbers:
        value = re.sub(r"\d+", " ", value)

    if keep_only_basic_punct:
        value = re.sub(r"[^a-zA-Zа-яА-Я0-9\s\.\,\-\+\#\/]", " ", value)

    value = _WS_RE.sub(" ", value).strip()
    return value[:max_length]
