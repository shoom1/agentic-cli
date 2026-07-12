import os

import pytest

from agentic_cli.file_utils import copy_regular_file_no_follow


def test_copies_regular_file(tmp_path):
    src = tmp_path / "a.txt"; src.write_bytes(b"hello")
    dst = tmp_path / "out" / "a.txt"
    copy_regular_file_no_follow(src, dst)
    assert dst.read_bytes() == b"hello"
    assert not dst.is_symlink()


def test_rejects_symlink_source(tmp_path):
    real = tmp_path / "real.txt"; real.write_bytes(b"secret")
    link = tmp_path / "link.txt"; link.symlink_to(real)
    with pytest.raises(OSError):  # O_NOFOLLOW -> ELOOP on the final component
        copy_regular_file_no_follow(link, tmp_path / "out.txt")


def test_rejects_fifo_source_without_hanging(tmp_path):
    import threading
    fifo = tmp_path / "pipe"; os.mkfifo(fifo)
    result = {}
    def run():
        try:
            copy_regular_file_no_follow(fifo, tmp_path / "out.txt")
        except (OSError, ValueError):
            result["rejected"] = True
    t = threading.Thread(target=run, daemon=True); t.start(); t.join(5)
    assert not t.is_alive(), "copy hung on a FIFO source (O_NONBLOCK missing?)"
    assert result.get("rejected")


def test_does_not_follow_dest_symlink(tmp_path):
    src = tmp_path / "src.txt"; src.write_bytes(b"NEW")
    outside = tmp_path / "outside.txt"; outside.write_bytes(b"ORIGINAL")
    deliver = tmp_path / "deliver"; deliver.mkdir()
    dst = deliver / "x.txt"; dst.symlink_to(outside)
    copy_regular_file_no_follow(src, dst)
    assert outside.read_bytes() == b"ORIGINAL"   # symlink target untouched
    assert dst.read_bytes() == b"NEW"            # dst replaced by a real file
    assert not dst.is_symlink()


def test_missing_source_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        copy_regular_file_no_follow(tmp_path / "nope.txt", tmp_path / "out.txt")
