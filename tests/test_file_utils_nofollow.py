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


def _open_dir(path):
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY)


def test_copies_between_directory_descriptors_after_both_are_swapped(tmp_path):
    # Once the directories are open, swapping either for a link elsewhere
    # changes nothing: the copy reads and writes the directories it opened.
    src_dir = tmp_path / "src"; src_dir.mkdir()
    (src_dir / "a.txt").write_bytes(b"INSIDE")
    dst_dir = tmp_path / "dst"; dst_dir.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "a.txt").write_bytes(b"OUTSIDE")
    src_fd, dst_fd = _open_dir(src_dir), _open_dir(dst_dir)
    try:
        for d in (src_dir, dst_dir):
            d.rename(d.with_name(d.name + ".parked"))
            d.symlink_to(outside, target_is_directory=True)
        copy_regular_file_no_follow("a.txt", "b.txt", src_dir_fd=src_fd, dst_dir_fd=dst_fd)
    finally:
        os.close(src_fd)
        os.close(dst_fd)
    copied = tmp_path / "dst.parked" / "b.txt"
    assert copied.read_bytes() == b"INSIDE"
    assert copied.stat().st_mode & 0o777 == 0o600
    assert sorted(p.name for p in outside.iterdir()) == ["a.txt"]
    assert sorted(p.name for p in (tmp_path / "dst.parked").iterdir()) == ["b.txt"]


def test_a_directory_descriptor_destination_is_a_name_not_a_path(tmp_path):
    src = tmp_path / "a.txt"; src.write_bytes(b"x")
    dst_dir = tmp_path / "dst"; (dst_dir / "sub").mkdir(parents=True)
    dst_fd = _open_dir(dst_dir)
    try:
        for bad in ("sub/a.txt", "..", ".", ""):
            with pytest.raises(ValueError):
                copy_regular_file_no_follow(src, bad, dst_dir_fd=dst_fd)
    finally:
        os.close(dst_fd)
    assert list((dst_dir / "sub").iterdir()) == []
