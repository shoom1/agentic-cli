"""Pages ``web_fetch`` saved into the project, for later ingestion.

Every successful fetch of a page the knowledge base can ingest is saved, byte
for byte, in ``./.<app_name>/fetched/``: ``<name><ext>`` holds the body as
received and ``<name>.meta.json`` records where it came from. ``<name>`` is
the first 16 hex digits of the SHA-256 of the requested URL, so fetching a
URL again replaces its pair. ``kb_ingest_file`` reads the metadata through
:func:`read_saved_page_meta`. Nothing here touches the network.

The folder is inside the project, which a repository controls: saving refuses
a folder that is a symlink, and cleanup deletes only regular files whose names
match the folder's own pattern.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import structlog

from agentic_cli.file_utils import atomic_write_bytes, atomic_write_json, atomic_write_text

logger = structlog.get_logger(__name__)

FOLDER_NAME = "fetched"
META_SUFFIX = ".meta.json"
META_VERSION = 1
# A metadata file is never bigger than this in normal use; anything past it
# (a planted file, for example) is treated as invalid without being parsed,
# so a pathological payload (deeply nested JSON, say) is never handed to
# json.loads in the first place.
MAX_META_BYTES = 64 * 1024

# Media type (lower-case, without parameters) -> extension of the saved page.
_EXTENSIONS = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "application/json": ".json",
    "text/xml": ".xml",
    "application/xml": ".xml",
    "application/pdf": ".pdf",
}
PAGE_EXTENSIONS = frozenset(_EXTENSIONS.values())
_OWN_FILE = re.compile(r"^([0-9a-f]{16})(\.meta\.json|\.html|\.txt|\.md|\.json|\.xml|\.pdf)$")


class SaveError(Exception):
    """A page could not be saved; the message says why."""


@dataclass(frozen=True)
class SavedPageMeta:
    """Where a saved page came from."""

    url: str
    final_url: str
    content_type: str
    charset: str | None
    fetched_at: datetime
    size: int
    truncated: bool


def saved_page_extension(content_type: str | None) -> str | None:
    """The extension a page of ``content_type`` is saved with; None if it is not saved."""
    media = (content_type or "").split(";", 1)[0].strip().lower()
    if media in _EXTENSIONS:
        return _EXTENSIONS[media]
    if media.endswith("+json"):
        return ".json"
    if media.endswith("+xml"):
        return ".xml"
    return None


def saved_pages_dir(app_name: str) -> Path:
    """The project's saved-pages folder, ``./.<app_name>/fetched``."""
    return Path.cwd() / f".{app_name}" / FOLDER_NAME


def page_name(url: str) -> str:
    """The name a URL's page and metadata files share."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _require_real_dir(path: Path) -> None:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise SaveError(f"{path} is a symlink; pages are not saved through it")
    if not stat.S_ISDIR(st.st_mode):
        raise SaveError(f"{path} is not a directory")


def _prepare_folder(folder: Path) -> None:
    """Create ``folder`` (0700) and its ``.gitignore``; refuse a symlinked one."""
    folder.parent.mkdir(parents=True, exist_ok=True)
    _require_real_dir(folder.parent)
    try:
        folder.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _require_real_dir(folder)
    gitignore = folder / ".gitignore"
    if not os.path.lexists(gitignore):
        atomic_write_text(gitignore, "*\n")


def _unlink_regular(path: Path) -> None:
    """Delete ``path`` if it is a regular file; never through a symlink."""
    try:
        if stat.S_ISREG(os.lstat(path).st_mode):
            path.unlink()
    except FileNotFoundError:
        pass


def save_page(
    folder: Path,
    *,
    url: str,
    final_url: str,
    content_type: str,
    charset: str | None,
    data: bytes,
    truncated: bool,
) -> Path:
    """Save a fetched page and its metadata; return the page's path.

    Raises:
        ValueError: pages of ``content_type`` are not saved.
        SaveError: the page could not be saved.
    """
    ext = saved_page_extension(content_type)
    if ext is None:
        raise ValueError(f"{content_type!r} pages are not saved")
    name = page_name(url)
    page = folder / f"{name}{ext}"
    try:
        _prepare_folder(folder)
        # Unlink the old metadata before writing the new page: otherwise a
        # refetch leaves a window where the new page sits beside the old,
        # still-valid metadata (wrong content type/charset/size until the new
        # metadata lands).
        _unlink_regular(folder / f"{name}{META_SUFFIX}")
        atomic_write_bytes(page, data)
        for other in PAGE_EXTENSIONS - {ext}:
            _unlink_regular(folder / f"{name}{other}")
        atomic_write_json(folder / f"{name}{META_SUFFIX}", {
            "version": META_VERSION,
            "url": url,
            "final_url": final_url,
            "content_type": content_type,
            "charset": charset,
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "size": len(data),
            "truncated": truncated,
        })
    except OSError as e:
        raise SaveError(f"could not save the page: {e.strerror or e}") from e
    return page


def is_saved_page_metadata(path: Path, folder: Path) -> bool:
    """Whether ``path`` is a metadata file in the saved-pages folder."""
    try:
        path = Path(path).resolve()
        return (
            path.parent == folder.resolve()
            and path.name.endswith(META_SUFFIX)
            and _OWN_FILE.match(path.name) is not None
        )
    except (OSError, ValueError):
        return False


def read_saved_page_meta(path: Path, folder: Path) -> SavedPageMeta | None:
    """The metadata of a page ``web_fetch`` saved, or None if ``path`` is not one.

    ``path`` is a saved page only when it sits directly in ``folder`` (and
    neither ``folder`` nor its parent is a symlink), has a page file's name,
    and has valid metadata beside it whose content type matches its extension.
    """
    try:
        path = Path(path).resolve()
        _require_real_dir(folder.parent)
        _require_real_dir(folder)
        if path.parent != folder.resolve():
            return None
        match = _OWN_FILE.match(path.name)
        if match is None or path.name.endswith(META_SUFFIX):
            return None
        meta_path = folder / f"{match.group(1)}{META_SUFFIX}"
        meta_stat = os.lstat(meta_path)
        if not stat.S_ISREG(meta_stat.st_mode):
            return None
        if meta_stat.st_size > MAX_META_BYTES:
            # Never parse a metadata file this large: a planted file with a
            # pathological payload (deeply nested JSON) must not even reach
            # json.loads.
            return None
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if data.get("version") != META_VERSION:
            return None
        url = str(data["url"])
        final_url = str(data.get("final_url") or url)
        if not final_url.startswith(("http://", "https://")):
            final_url = url
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        meta = SavedPageMeta(
            url=url,
            final_url=final_url,
            content_type=str(data["content_type"]),
            charset=str(data["charset"]) if data.get("charset") else None,
            fetched_at=fetched_at,
            size=int(data["size"]),
            truncated=bool(data.get("truncated", False)),
        )
    except Exception:
        # A planted or corrupt metadata file (bad JSON, a value nothing can
        # coerce, a payload pathological enough to blow the recursion limit,
        # ...) is not a saved page. It is never a bug in the caller, so
        # nothing escapes here.
        return None
    if saved_page_extension(meta.content_type) != path.suffix:
        return None
    if not meta.url.startswith(("http://", "https://")):
        return None
    return meta


def cleanup_saved_pages(folder: Path, *, max_age_days: float, max_bytes: int, keep: str) -> None:
    """Delete old saved pages.

    First every pair older than ``max_age_days``, then the oldest pairs until
    the folder holds at most ``max_bytes``. The pair named ``keep`` (a
    :func:`page_name`) is never deleted. Only regular files named like the
    folder's own are touched. Errors are logged, never raised.
    """
    try:
        _require_real_dir(folder.parent)
        _require_real_dir(folder)
        pairs: dict[str, list[tuple[Path, int, float]]] = {}
        with os.scandir(folder) as entries:
            for entry in entries:
                match = _OWN_FILE.match(entry.name)
                if match is None:
                    continue
                st = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(st.st_mode):
                    continue
                pairs.setdefault(match.group(1), []).append((Path(entry.path), st.st_size, st.st_mtime))

        now = time.time()
        saved_at = {name: _saved_at(files) for name, files in pairs.items()}
        for name in [n for n in pairs if n != keep and now - saved_at[n] > max_age_days * 86400]:
            _delete_pair(pairs.pop(name))

        total = sum(size for files in pairs.values() for _, size, _ in files)
        for name in sorted(pairs, key=saved_at.__getitem__):
            if total <= max_bytes:
                break
            if name == keep:
                continue
            total -= sum(size for _, size, _ in pairs[name])
            _delete_pair(pairs.pop(name))
    except Exception as e:
        # A planted or corrupt file anywhere in the folder (bad metadata,
        # a pathological payload, a permission error, ...) must never stop
        # cleanup from running, or stop web_fetch from reporting its result.
        logger.warning("saved_pages_cleanup_failed", folder=str(folder), error=str(e))


def _saved_at(files: list[tuple[Path, int, float]]) -> float:
    """When a pair was saved: its metadata's ``fetched_at``, else its newest file time."""
    for path, size, _ in files:
        if path.name.endswith(META_SUFFIX):
            if size <= MAX_META_BYTES:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    return datetime.fromisoformat(data["fetched_at"]).timestamp()
                except Exception:
                    pass
            break
    return max(mtime for _, _, mtime in files)


def _delete_pair(files: list[tuple[Path, int, float]]) -> None:
    for path, _, _ in files:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
