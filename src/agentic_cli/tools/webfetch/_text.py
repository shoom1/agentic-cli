"""Decoding a fetched body as text."""

from __future__ import annotations


def decode_body(data: bytes, charset: str | None) -> str:
    """Decode ``data`` with the response's declared ``charset``, else UTF-8.

    The label comes from the server. One Python cannot decode text with (an
    unknown name, a non-text codec such as ``base64``, or ``undefined``)
    falls back to UTF-8 instead of raising. Undecodable bytes are replaced.
    """
    if charset:
        try:
            return data.decode(charset, errors="replace")
        except (LookupError, UnicodeError):
            pass
    return data.decode("utf-8", errors="replace")
