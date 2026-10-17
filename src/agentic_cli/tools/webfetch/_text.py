"""Decoding a fetched body as text."""

from __future__ import annotations

from agentic_cli.tools._charset import text_codec


def decode_body(data: bytes, charset: str | None) -> str:
    """Decode ``data`` with the response's declared ``charset``, else UTF-8.

    The label comes from the server. One Python cannot decode text with (an
    unknown name, a non-text codec such as ``base64``, ``undefined``, or a
    real codec whose decode is pathologically slow such as ``punycode``)
    falls back to UTF-8 instead of raising or hanging. Undecodable bytes
    are replaced.
    """
    if charset:
        try:
            return data.decode(text_codec(charset), errors="replace")
        except (LookupError, UnicodeError):
            pass
    return data.decode("utf-8", errors="replace")
