"""``text_codec`` turns an untrusted charset label into a safe codec name.

A charset label reaches decoding unvalidated, from an HTTP ``Content-Type``
header or a planted saved-page metadata file. A few real codec names make
``bytes.decode`` run quadratically on ordinary page sizes (seconds for 200
KB, minutes for a few MB); ``text_codec`` denies those, and anything
``codecs.lookup`` does not resolve to an ordinary text codec.
"""

from __future__ import annotations

import pytest

from agentic_cli.tools._charset import text_codec


@pytest.mark.parametrize(
    "label,expected",
    [
        (None, "utf-8"),
        ("", "utf-8"),
        ("utf-8", "utf-8"),
        ("windows-1251", "cp1251"),
        ("no-such-codec", "utf-8"),
        ("base64", "utf-8"),
    ],
)
def test_text_codec(label, expected):
    assert text_codec(label) == expected


@pytest.mark.parametrize(
    "label",
    [
        "punycode",
        "PUNYCODE",
        "idna",
        "IDNA",
        "unicode_escape",
        "unicode-escape",
        "raw-unicode-escape",
        "raw_unicode_escape",
    ],
)
def test_denied_codecs_decode_as_utf8(label):
    assert text_codec(label) == "utf-8"
