"""Geometry helpers for PyGRID."""

from pathlib import Path


def read_extent(filename):
    """Read the four XY grid-corner coordinates from a TinyECL .ext file.

    TinyECL writes one corner per line.  A third value may be present; COORD
    only needs X and Y from this file.

    Expected corner order is:
        0: upper-left
        1: lower-left
        2: lower-right
        3: upper-right
    """

    path = Path(filename)
    points = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith(("*", "--", "#")):
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid coordinate in extent file '{path}' "
                    f"at line {line_number}: {line!r}"
                ) from exc

            points.append((x, y))

    if len(points) != 4:
        raise ValueError(
            f"Extent file '{path}' must contain exactly 4 corner points; "
            f"found {len(points)}."
        )

    return points


def create_pillars(corners, nx, ny):
    """Create the (nx+1)*(ny+1) horizontal pillar positions.

    The four boundary corners are bilinearly interpolated.  Pillars are
    returned in Eclipse COORD order: I changes fastest, then J.
    """

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must both be greater than zero")

    if len(corners) != 4:
        raise ValueError("corners must contain exactly four (x, y) points")

    nw, sw, se, ne = corners
    pillars = []

    for j in range(ny + 1):
        v = j / ny

        for i in range(nx + 1):
            u = i / nx

            x = (
                (1.0 - u) * (1.0 - v) * nw[0]
                + (1.0 - u) * v * sw[0]
                + u * v * se[0]
                + u * (1.0 - v) * ne[0]
            )

            y = (
                (1.0 - u) * (1.0 - v) * nw[1]
                + (1.0 - u) * v * sw[1]
                + u * v * se[1]
                + u * (1.0 - v) * ne[1]
            )

            pillars.append((x, y))

    return pillars
