"""The package's charset helper agrees with web_fetch's.

``agentic_cli.memory`` imports nothing from the rest of agentic-cli, so the
knowledge base's converter has its own copy of ``tools/_charset.py``, which
``web_fetch`` keeps using. The two must turn every label into the same codec.
"""

from __future__ import annotations

import pytest

from agentic_cli.memory.kb import _charset as package
from agentic_cli.tools import _charset as tools

LABELS = [
    None, "", "utf-8", "UTF8", "windows-1251", "cp1252", "latin-1", "iso-8859-1", "shift_jis",
    "no-such-codec", "base64", "rot13", "punycode", "PUNYCODE", "idna", "unicode_escape",
    "unicode-escape", "raw-unicode-escape", "raw_unicode_escape", "utf-16",
]


@pytest.mark.parametrize("label", LABELS)
def test_both_helpers_choose_the_same_codec(label):
    assert package.text_codec(label) == tools.text_codec(label)


def test_both_deny_the_same_codecs():
    assert package._DENIED_CODECS == tools._DENIED_CODECS
