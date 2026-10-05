"""Pages ``web_fetch`` saves into the project (``tools/webfetch/saved.py``).

Each page is saved byte for byte with a ``.meta.json`` beside it, in a folder
git ignores. The folder lives in the project, which a repository controls, so
saving refuses a symlinked folder and cleanup deletes only its own files.
"""

from __future__ import annotations

import json
import os
import stat
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agentic_cli.file_utils import atomic_write_bytes
from agentic_cli.tools.webfetch.saved import (
    MAX_META_BYTES,
    SaveError,
    cleanup_saved_pages,
    is_saved_page_metadata,
    page_name,
    read_saved_page_meta,
    save_page,
    saved_page_extension,
    saved_pages_dir,
)

URL = "https://example.com/guide"


@pytest.fixture
def folder(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return saved_pages_dir("app")


def _save(folder, url=URL, data=b"<p>guide</p>", content_type="text/html; charset=utf-8",
          charset="utf-8", final_url=None, truncated=False) -> Path:
    return save_page(
        folder, url=url, final_url=final_url or url, content_type=content_type,
        charset=charset, data=data, truncated=truncated,
    )


def _meta_path(folder, url=URL) -> Path:
    return folder / f"{page_name(url)}.meta.json"


def _age(folder, url, days):
    """Make a saved pair look ``days`` old."""
    path = _meta_path(folder, url)
    data = json.loads(path.read_text())
    then = datetime.now(timezone.utc) - timedelta(days=days)
    data["fetched_at"] = then.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(data))


def _pair_size(folder, url) -> int:
    name = page_name(url)
    return sum(p.stat().st_size for p in folder.iterdir() if p.name.startswith(name))


def test_atomic_write_bytes_writes_a_private_file(tmp_path):
    path = tmp_path / "page.pdf"
    atomic_write_bytes(path, b"%PDF-1.4\x00\xff")
    assert path.read_bytes() == b"%PDF-1.4\x00\xff"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_the_folder_is_in_the_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert saved_pages_dir("app") == Path.cwd() / ".app" / "fetched"


@pytest.mark.parametrize("content_type, ext", [
    ("text/html; charset=utf-8", ".html"),
    ("application/xhtml+xml", ".html"),
    ("text/plain", ".txt"),
    ("text/markdown", ".md"),
    ("application/json", ".json"),
    ("application/ld+json", ".json"),
    ("text/xml", ".xml"),
    ("application/rss+xml", ".xml"),
    ("application/pdf", ".pdf"),
])
def test_page_types_and_their_extensions(content_type, ext):
    assert saved_page_extension(content_type) == ext


@pytest.mark.parametrize("content_type", ["image/png", "application/zip", "application/octet-stream", "", None])
def test_other_types_are_not_saved(content_type):
    assert saved_page_extension(content_type) is None


def test_a_page_is_saved_with_its_metadata(folder):
    page = _save(folder, final_url="https://example.com/guide/")
    assert page == folder / f"{page_name(URL)}.html"
    assert page.read_bytes() == b"<p>guide</p>"
    meta = json.loads(_meta_path(folder).read_text())
    assert meta["version"] == 1
    assert meta["url"] == URL
    assert meta["final_url"] == "https://example.com/guide/"
    assert meta["content_type"] == "text/html; charset=utf-8"
    assert meta["charset"] == "utf-8"
    assert meta["size"] == len(b"<p>guide</p>")
    assert meta["truncated"] is False
    assert meta["fetched_at"].endswith("Z")
    for path in (page, _meta_path(folder)):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700


def test_the_folder_is_ignored_by_git(folder):
    _save(folder)
    assert (folder / ".gitignore").read_text() == "*\n"


def test_an_existing_gitignore_is_kept(folder):
    folder.mkdir(parents=True)
    (folder / ".gitignore").write_text("custom\n")
    _save(folder)
    assert (folder / ".gitignore").read_text() == "custom\n"


def test_a_refetch_replaces_the_pair_even_with_a_new_type(folder):
    _save(folder, data=b"<p>old</p>")
    page = _save(folder, data=b"%PDF-1.4 new", content_type="application/pdf", charset=None)
    assert page == folder / f"{page_name(URL)}.pdf"
    assert not (folder / f"{page_name(URL)}.html").exists()
    assert json.loads(_meta_path(folder).read_text())["content_type"] == "application/pdf"


def test_a_type_that_is_not_saved_raises(folder):
    with pytest.raises(ValueError):
        _save(folder, content_type="image/png")


@pytest.mark.parametrize("which", ["app", "fetched"])
def test_a_symlinked_folder_is_refused(tmp_path, monkeypatch, which):
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if which == "app":
        (tmp_path / ".app").symlink_to(elsewhere, target_is_directory=True)
    else:
        (tmp_path / ".app").mkdir()
        (tmp_path / ".app" / "fetched").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(SaveError, match="symlink"):
        _save(saved_pages_dir("app"))
    assert list(elsewhere.iterdir()) == []


def test_a_file_where_the_folder_should_be_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".app").mkdir()
    (tmp_path / ".app" / "fetched").write_text("not a folder")
    with pytest.raises(SaveError):
        _save(saved_pages_dir("app"))


def test_metadata_is_read_for_a_saved_page(folder):
    page = _save(folder, charset="windows-1251", truncated=True)
    meta = read_saved_page_meta(page, folder)
    assert meta is not None
    assert meta.url == URL
    assert meta.final_url == URL
    assert meta.content_type == "text/html; charset=utf-8"
    assert meta.charset == "windows-1251"
    assert meta.truncated is True
    assert meta.size == len(b"<p>guide</p>")
    assert meta.fetched_at.tzinfo is not None


def test_a_copy_outside_the_folder_is_not_a_saved_page(folder, tmp_path):
    page = _save(folder)
    copy = tmp_path / page.name
    copy.write_bytes(page.read_bytes())
    (tmp_path / _meta_path(folder).name).write_text(_meta_path(folder).read_text())
    assert read_saved_page_meta(copy, folder) is None


@pytest.mark.parametrize("damage", ["missing", "not json", "wrong version", "wrong type", "not http"])
def test_a_page_without_valid_metadata_is_not_a_saved_page(folder, damage):
    page = _save(folder)
    path = _meta_path(folder)
    data = json.loads(path.read_text())
    if damage == "missing":
        path.unlink()
    elif damage == "not json":
        path.write_text("{")
    else:
        if damage == "wrong version":
            data["version"] = 2
        elif damage == "wrong type":
            data["content_type"] = "application/pdf"
        else:
            data["url"] = "ftp://example.com/guide"
        path.write_text(json.dumps(data))
    assert read_saved_page_meta(page, folder) is None


def test_the_metadata_file_is_recognized(folder):
    page = _save(folder)
    assert is_saved_page_metadata(_meta_path(folder), folder) is True
    assert is_saved_page_metadata(page, folder) is False
    assert read_saved_page_meta(_meta_path(folder), folder) is None


def test_cleanup_deletes_pages_older_than_the_age_limit(folder):
    old = _save(folder, url="https://example.com/old")
    new = _save(folder, url="https://example.com/new")
    _age(folder, "https://example.com/old", 8)

    cleanup_saved_pages(folder, max_age_days=7, max_bytes=10**9, keep=page_name("https://example.com/new"))

    assert not old.exists()
    assert not _meta_path(folder, "https://example.com/old").exists()
    assert new.exists()


def test_cleanup_deletes_the_oldest_pages_over_the_size_limit(folder):
    urls = [f"https://example.com/{i}" for i in range(4)]
    pages = [_save(folder, url=u, data=b"x" * 1000) for u in urls]
    for i, u in enumerate(urls):
        _age(folder, u, days=4 - i)  # urls[0] is the oldest
    limit = _pair_size(folder, urls[2]) + _pair_size(folder, urls[3])

    cleanup_saved_pages(folder, max_age_days=30, max_bytes=limit, keep=page_name(urls[3]))

    assert [p.exists() for p in pages] == [False, False, True, True]


def test_cleanup_never_deletes_the_page_just_saved(folder):
    page = _save(folder)
    _age(folder, URL, 30)
    cleanup_saved_pages(folder, max_age_days=7, max_bytes=1, keep=page_name(URL))
    assert page.exists()
    assert _meta_path(folder).exists()


def test_cleanup_touches_only_its_own_regular_files(folder, tmp_path):
    _save(folder, url="https://example.com/old")
    _age(folder, "https://example.com/old", 30)
    kept = _save(folder, url="https://example.com/new")
    long_ago = time.time() - 60 * 86400
    notes = folder / "notes.txt"
    notes.write_text("mine")
    os.utime(notes, (long_ago, long_ago))
    outside = tmp_path / "outside.html"
    outside.write_text("outside")
    link = folder / f"{'0' * 16}.html"
    link.symlink_to(outside)
    os.utime(link, (long_ago, long_ago), follow_symlinks=False)

    cleanup_saved_pages(folder, max_age_days=7, max_bytes=1, keep=page_name("https://example.com/new"))

    assert notes.exists()
    assert (folder / ".gitignore").exists()
    assert link.is_symlink() and outside.read_text() == "outside"
    assert kept.exists()
    assert not (folder / f"{page_name('https://example.com/old')}.html").exists()


def test_a_page_without_metadata_ages_by_its_file_time(folder):
    page = _save(folder, url="https://example.com/orphan")
    _meta_path(folder, "https://example.com/orphan").unlink()
    long_ago = time.time() - 10 * 86400
    os.utime(page, (long_ago, long_ago))

    cleanup_saved_pages(folder, max_age_days=7, max_bytes=10**9, keep="f" * 16)

    assert not page.exists()


def test_cleanup_does_nothing_in_a_symlinked_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    victim = elsewhere / f"{'a' * 16}.html"
    victim.write_text("old")
    long_ago = time.time() - 60 * 86400
    os.utime(victim, (long_ago, long_ago))
    (tmp_path / ".app").mkdir()
    (tmp_path / ".app" / "fetched").symlink_to(elsewhere, target_is_directory=True)

    cleanup_saved_pages(saved_pages_dir("app"), max_age_days=7, max_bytes=1, keep="b" * 16)

    assert victim.exists()


def test_the_cleanup_settings():
    from agentic_cli.config import BaseSettings
    from agentic_cli.settings_persistence import PROJECT_SETTABLE_KEYS

    settings = BaseSettings()
    assert settings.webfetch_saved_max_age_days == 7
    assert settings.webfetch_saved_max_mb == 200
    assert {"webfetch_saved_max_age_days", "webfetch_saved_max_mb"} <= PROJECT_SETTABLE_KEYS


def test_negative_saved_page_settings_are_rejected():
    from pydantic import ValidationError

    from agentic_cli.config import BaseSettings

    with pytest.raises(ValidationError):
        BaseSettings(webfetch_saved_max_age_days=-1)
    with pytest.raises(ValidationError):
        BaseSettings(webfetch_saved_max_mb=-1)


# -- I-1: a planted metadata file must never make cleanup or a read raise. --


def test_this_pythons_json_blows_the_recursion_limit_on_deeply_nested_input():
    """Confirms the probe payload actually triggers RecursionError here,
    before relying on that in the tests below."""
    nested = "[" * 100_000 + "]" * 100_000
    with pytest.raises(RecursionError):
        json.loads(nested)


def test_cleanup_survives_a_planted_deeply_nested_json_metadata_file(folder):
    old = _save(folder, url="https://example.com/old")
    _age(folder, "https://example.com/old", 30)
    planted = folder / f"{'f' * 16}{'.meta.json'}"
    planted.write_text("[" * 100_000 + "]" * 100_000)

    cleanup_saved_pages(folder, max_age_days=7, max_bytes=10**9, keep="0" * 16)

    assert not old.exists()
    assert not _meta_path(folder, "https://example.com/old").exists()


def test_read_saved_page_meta_is_none_for_a_pathological_size(folder):
    page = _save(folder)
    path = _meta_path(folder)
    data = json.loads(path.read_text())
    data["size"] = 1e999
    path.write_text(json.dumps(data))

    assert read_saved_page_meta(page, folder) is None


def test_read_saved_page_meta_is_none_for_an_oversized_metadata_file(folder):
    page = _save(folder)
    path = _meta_path(folder)
    data = json.loads(path.read_text())
    data["padding"] = "x" * (MAX_META_BYTES + 1)
    assert len(json.dumps(data)) > MAX_META_BYTES
    path.write_text(json.dumps(data))

    assert read_saved_page_meta(page, folder) is None


# -- M-3: final_url is validated like url; a naive fetched_at is UTC. --


def test_a_non_http_final_url_falls_back_to_url(folder):
    page = _save(folder)
    path = _meta_path(folder)
    data = json.loads(path.read_text())
    data["final_url"] = "ftp://example.com/guide"
    path.write_text(json.dumps(data))

    meta = read_saved_page_meta(page, folder)

    assert meta is not None
    assert meta.final_url == URL


def test_a_naive_fetched_at_is_treated_as_utc(folder):
    page = _save(folder)
    path = _meta_path(folder)
    data = json.loads(path.read_text())
    data["fetched_at"] = "2026-01-01T00:00:00"
    path.write_text(json.dumps(data))

    meta = read_saved_page_meta(page, folder)

    assert meta is not None
    assert meta.fetched_at == datetime(2026, 1, 1, tzinfo=timezone.utc)


# -- M-7: a refetch/gitignore write never goes through a planted symlink. --


def test_a_refetch_leaves_a_symlinked_other_extension_alone(folder, tmp_path):
    _save(folder, data=b"<p>old</p>")  # saved as .html
    outside = tmp_path / "outside.txt"
    outside.write_text("do not touch")
    link = folder / f"{page_name(URL)}.txt"
    link.symlink_to(outside)

    page = _save(folder, data=b"%PDF-1.4 new", content_type="application/pdf", charset=None)

    assert page == folder / f"{page_name(URL)}.pdf"
    assert link.is_symlink()
    assert outside.read_text() == "do not touch"


def test_a_symlinked_gitignore_is_not_written_through(folder, tmp_path):
    folder.mkdir(parents=True)
    outside = tmp_path / "outside-gitignore"
    outside.write_text("do not touch")
    (folder / ".gitignore").symlink_to(outside)

    _save(folder)

    assert (folder / ".gitignore").is_symlink()
    assert outside.read_text() == "do not touch"
