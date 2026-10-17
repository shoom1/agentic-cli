"""A safe text codec for an untrusted charset label.

The label reaches decoding from an HTTP ``Content-Type`` header (the
fetcher) or a planted saved-page metadata file — never validated. A
handful of real codec names (``punycode``, ``idna``, the escape codecs)
make ``bytes.decode`` run quadratically in the input size: a 200 KB page
takes under a second, 1 MB takes tens of seconds, and a multi-megabyte
page runs for minutes, synchronously. ``text_codec`` turns any label that
is not an ordinary, fast text codec into ``"utf-8"`` before it ever
reaches ``decode``.
"""

from __future__ import annotations

import codecs

# Real text codecs codecs.lookup resolves, but whose decode is
# pathologically slow on ordinary page sizes (see module docstring).
# Never used even though they are valid text encodings.
_DENIED_CODECS = frozenset({"punycode", "idna", "unicode-escape", "raw-unicode-escape"})


def text_codec(label: str | None) -> str:
    """The codec name to decode with, given an untrusted charset ``label``.

    Returns ``"utf-8"`` when ``label`` is ``None``/empty, unknown to
    ``codecs.lookup``, not a text codec (for example ``base64``), or one
    of the denied codecs above. Otherwise returns the codec's normalized
    name (``codecs.lookup(label).name``).
    """
    if not label:
        return "utf-8"
    try:
        info = codecs.lookup(label)
    except LookupError:
        return "utf-8"
    if getattr(info, "_is_text_encoding", True) is False:
        return "utf-8"
    if info.name in _DENIED_CODECS:
        return "utf-8"
    return info.name
