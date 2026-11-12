"""Backup safety helpers for ProjectManager.

- Credentials never reach a backup copy or a loaded project state (01-L6).
- The auto-backup mirror keeps rotated generations of the files a sync
  replaces or removes (18-M6).
- Zip restores are capped in member count and uncompressed size (18-L6).
"""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from gui.settings_persistence import API_KEY_QSETTINGS_KEYS

logger = logging.getLogger(__name__)

# JSON project files that hold a config mapping; backup copies of them are
# written without credential entries.
CREDENTIAL_SCRUBBED_FILES = frozenset({"project_state.json", "pipeline_config.json"})

AUTO_BACKUP_GENERATIONS = 5
GENERATIONS_DIR = "generations"

MAX_RESTORE_MEMBERS = 200_000
MAX_RESTORE_BYTES = 64 * 1024 ** 3


def is_credential_field(name: str) -> bool:
    """API keys and tokens; ``pubmed_email`` is a contact address, not a secret."""
    return name.endswith("_api_key") or (
        name in API_KEY_QSETTINGS_KEYS and name != "pubmed_email"
    )


def strip_credentials(mapping: dict) -> tuple[dict, list[str]]:
    """(mapping without credential keys, sorted names of the keys left out)."""
    kept = {k: v for k, v in mapping.items() if not is_credential_field(str(k))}
    dropped = sorted(str(k) for k in mapping if is_credential_field(str(k)))
    return kept, dropped


def scrub_project_state(state: dict) -> tuple[dict, list[str]]:
    """Return ``state`` without credentials at its top level or in ``config``."""
    clean, dropped = strip_credentials(state)
    config = clean.get("config")
    if isinstance(config, dict):
        clean_config, more = strip_credentials(config)
        clean = {**clean, "config": clean_config}
        dropped = dropped + more
    return clean, dropped


def scrubbed_json_bytes(path: Path) -> bytes | None:
    """Credential-free JSON for a scrubbed project file.

    Returns None when ``path`` is not such a file, cannot be parsed, or holds
    no credentials, in which case the file is copied unchanged.
    """
    if path.name not in CREDENTIAL_SCRUBBED_FILES:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    clean, dropped = scrub_project_state(data)
    if not dropped:
        return None
    logger.warning(
        "Backup copy of %s leaves out stored credentials: %s",
        path, ", ".join(dropped),
    )
    return json.dumps(clean, indent=2).encode("utf-8")


def write_backup_member(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    """Add ``path`` to a backup zip, without credentials when it is a config file."""
    scrubbed = scrubbed_json_bytes(path)
    if scrubbed is None:
        zf.write(path, arcname)
    else:
        zf.writestr(arcname, scrubbed)


def _write_atomic(dest: Path, data: bytes) -> None:
    temp = dest.with_name(dest.name + ".tmp")
    temp.write_bytes(data)
    temp.replace(dest)


class GenerationKeeper:
    """Moves files a sync would replace or delete into a timestamped generation.

    One generation folder is created lazily per sync under
    ``<container>/generations/<timestamp>/`` and holds the previous version
    of every changed or removed file, at its project-relative path.
    """

    def __init__(self, backup_root: Path) -> None:
        self.backup_root = backup_root
        self.generations_root = backup_root.parent / GENERATIONS_DIR
        self._current: Path | None = None
        self.preserved = 0

    def _generation(self) -> Path:
        if self._current is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._current = self.generations_root / stamp
        return self._current

    def preserve(self, existing: Path) -> None:
        """Move an existing mirror file into this sync's generation folder."""
        if not existing.is_file():
            return
        target = self._generation() / existing.relative_to(self.backup_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing.replace(target)
        self.preserved += 1

    def prune(self, keep: int | None = None) -> None:
        """Delete all but the newest ``keep`` generation folders."""
        keep = AUTO_BACKUP_GENERATIONS if keep is None else keep
        if not self.generations_root.is_dir():
            return
        folders = sorted(p for p in self.generations_root.iterdir() if p.is_dir())
        for old in folders[:max(0, len(folders) - keep)]:
            shutil.rmtree(old, ignore_errors=True)
            if old.exists():
                logger.warning("Could not remove old backup generation: %s", old)


def _file_needs_copy(src: Path, dest: Path) -> bool:
    if not dest.exists():
        return True
    try:
        src_stat = src.stat()
        dest_stat = dest.stat()
    except OSError:
        return True
    if src_stat.st_size != dest_stat.st_size:
        return True
    return src_stat.st_mtime_ns > dest_stat.st_mtime_ns


def mirror_file(src: Path, dest: Path, keeper: GenerationKeeper) -> bool:
    """Copy ``src`` into the mirror when changed; True when ``dest`` was written.

    The previous version of ``dest`` moves into the current generation, and
    config files are written without credentials.
    """
    scrubbed = scrubbed_json_bytes(src)
    if scrubbed is not None:
        if dest.exists() and dest.read_bytes() == scrubbed:
            return False
        keeper.preserve(dest)
        _write_atomic(dest, scrubbed)
        return True
    if not _file_needs_copy(src, dest):
        return False
    keeper.preserve(dest)
    temp_dest = dest.with_name(dest.name + ".tmp")
    shutil.copy2(src, temp_dest)
    temp_dest.replace(dest)
    return True


def check_zip_limits(zf: zipfile.ZipFile) -> None:
    """Refuse archives with too many members or too much uncompressed data."""
    members = zf.infolist()
    if len(members) > MAX_RESTORE_MEMBERS:
        raise ValueError(
            f"Backup has {len(members)} files; the limit is {MAX_RESTORE_MEMBERS}."
        )
    total = sum(member.file_size for member in members)
    if total > MAX_RESTORE_BYTES:
        raise ValueError(
            f"Backup expands to {total} bytes; the limit is {MAX_RESTORE_BYTES}."
        )
