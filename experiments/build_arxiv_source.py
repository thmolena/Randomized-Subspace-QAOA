"""Build a deterministic, self-contained arXiv source archive for the paper."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

from validate_manuscript import ARCHIVE, PAPER, archive_sources, validate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_archive() -> dict[str, object]:
    preflight = validate(require_archive=False)
    sources = archive_sources()
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("arXiv source files are missing: " + ", ".join(missing))

    temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as handle:
        for name in sorted(sources):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            handle.writestr(info, sources[name].read_bytes(), compresslevel=9)
    os.replace(temporary, ARCHIVE)
    return {
        "archive": str(ARCHIVE.relative_to(PAPER.parent)),
        "members": len(sources),
        "main_display_items": preflight["main_display_items"],
        "visible_references": preflight["unique_cited_works"],
        "sha256": _sha256(ARCHIVE),
    }


def main() -> None:
    print(json.dumps(build_archive(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
