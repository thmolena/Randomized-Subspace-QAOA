#!/usr/bin/env python3
"""Build and verify the deterministic 22-file RSQ PRX/arXiv source archive."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile

from validate_manuscript import ARCHIVE, PAPER, archive_sources, validate


def build_archive() -> dict[str, object]:
    source_report = validate(require_archive=False)
    sources = archive_sources()
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("arXiv source files are missing: " + ", ".join(missing))

    temporary = ARCHIVE.with_suffix(ARCHIVE.suffix + ".tmp")
    try:
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
    finally:
        if temporary.exists():
            temporary.unlink()

    archive_report = validate(require_archive=True)
    return {
        "archive": str(ARCHIVE.relative_to(PAPER.parent)),
        "members": archive_report["archive_members"],
        "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "source_validated": source_report["main_sha256"],
        "archive_validated": archive_report["archive_checked"],
    }


def main() -> None:
    print(json.dumps(build_archive(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
