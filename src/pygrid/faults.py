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
    """Collection of TinyECL top/bottom fault traces.

    ``traces`` are the top traces from ``*.flt`` and are the authoritative
    source for Eclipse FAULTS names and fault topology.

    ``bottom_traces`` are the optional ``*.flb`` traces.  A fault is slanted
    only when the same fault name occurs in both files.  Bottom traces never
    rename top traces and never create synthetic names.
    """

    def __init__(self, traces=None, bottom_traces=None):
        self.traces = list(traces or [])
        self.bottom_traces = list(bottom_traces or [])

    def __bool__(self):
        return bool(self.traces)

    @property
    def segments(self):
        result = []
        for trace in self.traces:
            for p1, p2 in trace.segments:
                result.append((trace, np.asarray(p1, float), np.asarray(p2, float)))
        return result

    @property
    def names(self):
        """Top-fault names in first-occurrence order."""
        return _unique_names(self.traces)

    @property
    def bottom_names(self):
        """Bottom-fault names in first-occurrence order."""
        return _unique_names(self.bottom_traces)

    @property
    def slanted_names(self):
        """Fault names present in both ``*.flt`` and ``*.flb``."""
        bottom = set(self.bottom_names)
        return [name for name in self.names if name in bottom]

    @classmethod
    def from_tinyecl(cls, top_filename, bottom_filename=None):
        """Read TinyECL top and optional bottom fault polylines.

        TinyECL stores the fault name in the final column of both ``*.flt``
        and ``*.flb`` records.  The name in ``*.flt`` is authoritative.

        Older PyGRID code matched contiguous top groups to bottom groups by
        *order* and generated fallback names such as ``FAULT_4`` when the
        counts differed.  That is wrong for TinyECL because:

        * vertical faults may exist only in ``*.flt``;
        * a fault name may occur in more than one non-contiguous polyline;
        * ``*.flb`` contains only the faults that have a bottom trace.

        We therefore preserve every source name exactly as written and pair
        slanted geometry only by matching names.
        """
        top_groups = _read_contiguous_groups(top_filename, id_column=-1)
        traces = [
            FaultTrace(
                name=_clean_fault_name(source_id),
                source_id=_clean_fault_name(source_id),
                points=np.asarray(points, dtype=float),
            )
            for source_id, points in top_groups
        ]

        bottom_traces = []
        if bottom_filename and Path(bottom_filename).exists():
            bottom_groups = _read_contiguous_groups(bottom_filename, id_column=-1)
            bottom_traces = [
                FaultTrace(
                    name=_clean_fault_name(source_id),
                    source_id=_clean_fault_name(source_id),
                    points=np.asarray(points, dtype=float),
                )
                for source_id, points in bottom_groups
            ]

        return cls(traces=traces, bottom_traces=bottom_traces)

    def named_traces(self, name, bottom=False):
        """Return all top or bottom polyline groups with ``name``."""
        source = self.bottom_traces if bottom else self.traces
        return [trace for trace in source if trace.name == name]

    def slant_displacement(self, name, point):
        """Return local XY top-to-bottom displacement for a slanted fault.

        The displacement is measured geometrically, not by record order:

        1. find the nearest point on any top polyline with ``name``;
        2. find the nearest point on any bottom polyline with the same name;
        3. return ``bottom - top``.

        Nearest-point matching is appropriate for TinyECL's sliding-fault
        input because the bottom trace is normally a laterally displaced
        version of the same fault.  It also works when one fault name occurs
        in several non-contiguous top polyline groups.
        """
        point = np.asarray(point, dtype=float)
        top = self.named_traces(name, bottom=False)
        bottom = self.named_traces(name, bottom=True)
        if not top or not bottom:
            return np.zeros(2, dtype=float)

        top_point = _nearest_point_on_traces(point, top)
        bottom_point = _nearest_point_on_traces(top_point, bottom)
        return bottom_point - top_point

    def blocks_segment(self, p1, p2):
        """Return True if the open line p1-p2 crosses any top fault segment."""
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


def _clean_fault_name(value):
    """Normalize only surrounding whitespace/quotes; preserve the source name."""
    return str(value).strip().strip("'\"")


def _unique_names(traces):
    seen = set()
    result = []
    for trace in traces:
        if trace.name not in seen:
            seen.add(trace.name)
            result.append(trace.name)
    return result


def _nearest_point_on_segment(point, a, b):
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= _EPS:
        return a.copy()
    t = float(np.dot(point - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    return a + t * ab


def _nearest_point_on_traces(point, traces):
    point = np.asarray(point, dtype=float)
    best = None
    best_d2 = np.inf

    for trace in traces:
        if len(trace.points) == 1:
            candidate = np.asarray(trace.points[0], dtype=float)
            d2 = float(np.sum((candidate - point) ** 2))
            if d2 < best_d2:
                best = candidate
                best_d2 = d2
            continue

        for a, b in trace.segments:
            candidate = _nearest_point_on_segment(point, a, b)
            d2 = float(np.sum((candidate - point) ** 2))
            if d2 < best_d2:
                best = candidate
                best_d2 = d2

    if best is None:
        return point.copy()
    return np.asarray(best, dtype=float)


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

            ident = _clean_fault_name(parts[id_column])
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
