"""Create a deterministic public archive, excluding local environments and references."""

import argparse
import hashlib
import zipfile
from pathlib import Path

from check_repo import ROOT, publication_files, validate


def package(destination):
    report = validate()
    if not report["ok"]:
        raise ValueError(report["errors"])
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    hashes = []
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in publication_files():
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            info = zipfile.ZipInfo(
                "microduck-room-search/" + relative, date_time=(2026, 11, 6, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if path.name == "run.sh" else 0o100644) << 16
            archive.writestr(info, content)
            hashes.append(hashlib.sha256(content).hexdigest() + "  " + relative)
        info = zipfile.ZipInfo(
            "microduck-room-search/MANIFEST.sha256", date_time=(2026, 11, 6, 0, 0, 0)
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, "\n".join(hashes) + "\n")
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        checksum + "  " + destination.name + "\n"
    )
    print(destination, destination.stat().st_size, checksum)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", type=Path, default=ROOT.parent / "microduck-room-search-v0.2.0.zip"
    )
    package(p.parse_args().output)
