"""Fault geometry and TinyECL fault-file handling."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_EPS = 1.0e-9


@dataclass
class FaultTrace:
    """One polyline fault trace."""

    name: str
    points: np.ndarray
    source_id: str = ""

    @property
    def segments(self):
        if len(self.points) < 2:
            return []
        return list(zip(self.points[:-1], self.points[1:]))


class FaultSet:
    """Collection of fault traces used as interpolation barriers."""

    def __init__(self, traces=None):
        self.traces = list(traces or [])

    def __bool__(self):
        return bool(self.traces)

    @property
    def segments(self):
        result = []
        for trace in self.traces:
            for p1, p2 in trace.segments:
                result.append((trace, np.asarray(p1, float), np.asarray(p2, float)))
        return result

    @classmethod
    def from_tinyecl(cls, top_filename, names_filename=None):
        """
        Read TinyECL fault polylines.

        ``.flt`` files contain X Y and a polyline identifier.  The optional
        ``.flb`` file contains the same traces with a final textual fault name.
        The contiguous polyline groups in both files are matched by order.
        """
        top_groups = _read_contiguous_groups(top_filename, id_column=2)

        names = []
        if names_filename and Path(names_filename).exists():
            name_groups = _read_contiguous_groups(names_filename, id_column=-1)
            names = [group_id for group_id, _ in name_groups]

        traces = []
        for idx, (source_id, points) in enumerate(top_groups):
            name = names[idx] if idx < len(names) else f"FAULT_{idx + 1}"
            traces.append(
                FaultTrace(
                    name=name,
                    source_id=str(source_id),
                    points=np.asarray(points, dtype=float),
                )
            )

        return cls(traces)

    def blocks_segment(self, p1, p2):
        """Return True if the open line p1-p2 crosses any fault segment."""
        return bool(self.blocked_mask(p1, np.atleast_2d(p2))[0])

    def blocked_mask(self, origin, targets):
        """Vectorized barrier test from one origin to many target points."""
        origin = np.asarray(origin, dtype=float)
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        blocked = np.zeros(len(targets), dtype=bool)

        a = origin
        b = targets
        ab = b - a

        for _, c, d in self.segments:
            cd = d - c

            o1 = ab[:, 0] * (c[1] - a[1]) - ab[:, 1] * (c[0] - a[0])
            o2 = ab[:, 0] * (d[1] - a[1]) - ab[:, 1] * (d[0] - a[0])

            ca = a - c
            cb = b - c
            o3 = cd[0] * ca[1] - cd[1] * ca[0]
            o4 = cd[0] * cb[:, 1] - cd[1] * cb[:, 0]

            crosses = (
                (((o1 > _EPS) & (o2 < -_EPS)) | ((o1 < -_EPS) & (o2 > _EPS)))
                & (((o3 > _EPS) & (o4 < -_EPS)) | ((o3 < -_EPS) & (o4 > _EPS)))
            )
            blocked |= crosses

        return blocked


def _read_contiguous_groups(filename, id_column):
    groups = []
    current_id = None
    current_points = []

    with open(Path(filename), "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*") or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            ident = parts[id_column]
            point = (float(parts[0]), float(parts[1]))

            if current_id is None:
                current_id = ident
            elif ident != current_id:
                groups.append((current_id, current_points))
                current_id = ident
                current_points = []

            current_points.append(point)

    if current_id is not None and current_points:
        groups.append((current_id, current_points))

    return groups


def _orientation(a, b, c):
    return float(np.cross(b - a, c - a))


def proper_segment_intersection(a, b, c, d, eps=_EPS):
    """
    Test whether segments AB and CD cross in their interiors.

    Touching only at an endpoint is not treated as crossing.  This is useful
    for interpolation queries nudged onto one side of a fault.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    d = np.asarray(d, float)

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    return (
        ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
        and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
    )


def segment_intersection(a, b, c, d, eps=_EPS):
    """Return the intersection point for two finite segments, else None."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    d = np.asarray(d, float)

    r = b - a
    s = d - c
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) <= eps:
        return None

    ca = c - a
    t = (ca[0] * s[1] - ca[1] * s[0]) / denom
    u = (ca[0] * r[1] - ca[1] * r[0]) / denom

    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        return a + t * r
    return None
