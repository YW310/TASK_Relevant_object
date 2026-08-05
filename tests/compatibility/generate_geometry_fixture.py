"""Generate the deterministic NumPy NPZ used by compatibility fixtures."""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path


OUTPUT = (
    Path(__file__).with_name("fixtures")
    / "fusion" / "frames" / "000000_0" / "fused_geometry.npz"
)


def _npy_float32_2x3(values: tuple[float, ...]) -> bytes:
    if len(values) != 6:
        raise ValueError("The fixture array must contain exactly six float values.")
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"
    preamble_length = 10
    padding = 16 - ((preamble_length + len(header) + 1) % 16)
    encoded_header = (header + (" " * padding) + "\n").encode("latin-1")
    return (
        b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(encoded_header))
        + encoded_header + struct.pack("<6f", *values)
    )


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    archive_buffer = io.BytesIO()
    info = zipfile.ZipInfo("O1/points_world.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(
            info,
            _npy_float32_2x3((0.0, 0.0, 1.0, 0.01, 0.0, 1.0)),
        )
    OUTPUT.write_bytes(archive_buffer.getvalue())


if __name__ == "__main__":
    main()

