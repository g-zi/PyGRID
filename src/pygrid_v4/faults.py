"""Fault geometry and TinyECL fault-file handling."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_EPS = 1.0e-9


@dataclass
class FaultTrace:
    """One polyline fault trace.

    ``surface_barrier`` controls only structural-surface interpolation and
    smoothing.  A trace may be transparent to the surface while still being
    emitted as an Eclipse FAULTS face and still splitting grid connections.
    This is how PyGRID represents a zero-throw fault.
    """

    name: str
    points: np.ndarray
    source_id: str = ""
    surface_barrier: bool = True
    surface_crossings: tuple = ()

    @property
    def segments(self):
        if len(self.points) < 2:
            return []
        return list(zip(self.points[:-1], self.points[1:]))

    @property
    def length(self):
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


class FaultSet:
    """Collection of TinyECL top/bottom fault traces.

    ``traces`` are the top traces from ``*.flt`` and are authoritative for
    Eclipse FAULTS names and topology.

    ``bottom_traces`` are optional ``*.flb`` traces.  They add slanted-fault
    geometry only where a same-name bottom polyline can be matched to a
    particular same-name top polyline.  A bottom trace never renames a fault
    and never creates a synthetic fault name.
    """

    def __init__(self, traces=None, bottom_traces=None):
        self.traces = list(traces or [])
        self.bottom_traces = list(bottom_traces or [])
        self._matched_pairs_cache = {}

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
    def surface_barrier_segments(self):
        """Top-fault segments that may block structural interpolation."""
        result = []
        for trace in self.traces:
            if not trace.surface_barrier:
                continue
            for p1, p2 in trace.segments:
                result.append((trace, np.asarray(p1, float), np.asarray(p2, float)))
        return result

    @property
    def zero_throw_traces(self):
        """Top traces classified as surface-continuous / zero throw."""
        return [trace for trace in self.traces if not trace.surface_barrier]

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
        """Fault names having at least one matched top/bottom polyline pair."""
        return [name for name in self.names if self.matched_trace_pairs(name)]

    @classmethod
    def from_tinyecl(cls, top_filename, bottom_filename=None):
        """Read TinyECL top and optional bottom fault polylines.

        TinyECL stores the fault name in the final column of both ``*.flt``
        and ``*.flb`` records.  The name in ``*.flt`` is authoritative.

        Important rules:

        * vertical faults may exist only in ``*.flt``;
        * a fault name may occur in several non-contiguous top polylines;
        * ``*.flb`` may contain only part of a same-name fault;
        * top/bottom groups are paired geometrically, never by file order;
        * no ``FAULT_n`` fallback names are ever generated.
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

    def matched_trace_pairs(self, name):
        """Return one-to-one same-name top/bottom polyline pairs.

        TinyECL may write a repeated fault name in several non-contiguous
        ``*.flt`` groups while ``*.flb`` contains only one of those pieces.
        Pairing all same-name top groups to the same bottom trace would make
        unrelated grid pillars converge toward that bottom polyline and create
        long spikes.  We therefore pair contiguous groups one-to-one by a
        simple geometric endpoint/length cost.

        Each returned item is ``(top_trace, bottom_trace)``.  Bottom-point
        order is reversed when that gives the better endpoint match.
        """
        if name in self._matched_pairs_cache:
            return self._matched_pairs_cache[name]

        tops = self.named_traces(name, bottom=False)
        bottoms = self.named_traces(name, bottom=True)
        if not tops or not bottoms:
            self._matched_pairs_cache[name] = []
            return []

        candidates = []
        for ti, top in enumerate(tops):
            for bi, bottom in enumerate(bottoms):
                same = (
                    float(np.linalg.norm(top.points[0] - bottom.points[0]))
                    + float(np.linalg.norm(top.points[-1] - bottom.points[-1]))
                )
                reverse = (
                    float(np.linalg.norm(top.points[0] - bottom.points[-1]))
                    + float(np.linalg.norm(top.points[-1] - bottom.points[0]))
                )
                reverse_bottom = reverse < same
                endpoint_cost = min(same, reverse)
                length_cost = abs(top.length - bottom.length)
                cost = endpoint_cost + 0.25 * length_cost
                candidates.append((cost, ti, bi, reverse_bottom))

        used_top = set()
        used_bottom = set()
        pairs = []

        for _cost, ti, bi, reverse_bottom in sorted(candidates):
            if ti in used_top or bi in used_bottom:
                continue

            used_top.add(ti)
            used_bottom.add(bi)

            bottom = bottoms[bi]
            if reverse_bottom:
                bottom = FaultTrace(
                    name=bottom.name,
                    source_id=bottom.source_id,
                    points=np.asarray(bottom.points[::-1], dtype=float),
                )

            pairs.append((tops[ti], bottom))

        self._matched_pairs_cache[name] = pairs
        return pairs

    def classify_surface_barriers(
        self,
        contour_map,
        min_crossings=2,
        min_station_separation=0.15,
    ):
        """Classify zero-throw fault pieces from contour crossings.

        TinyECL contour lines are interpreted as XYZ point clouds.  If two or
        more *different contour levels* cross the interior of one contiguous
        top-fault polyline, well separated along that polyline, the structural
        interpretation itself says that the horizon is continuous across that
        fault piece.  Such a trace remains a grid fault but is not used as a
        hard surface-interpolation/smoothing barrier.

        Matched top/bottom (slanted) fault pieces are left as barriers because
        their paired geometry is an explicit 3-D fault interpretation.
        """
        min_crossings = int(min_crossings)
        min_station_separation = float(min_station_separation)
        if min_crossings < 2:
            raise ValueError("min_crossings must be at least 2")
        if not 0.0 <= min_station_separation <= 1.0:
            raise ValueError("min_station_separation must be between 0 and 1")

        for trace in self.traces:
            trace.surface_barrier = True
            trace.surface_crossings = ()

        source_lines = getattr(contour_map, "_source_lines", None)
        if not source_lines:
            return

        matched_top_ids = {
            id(top)
            for name in self.names
            for top, _bottom in self.matched_trace_pairs(name)
        }

        for trace in self.traces:
            if id(trace) in matched_top_ids:
                continue

            hits_by_z = {}

            for source_line in source_lines:
                if len(source_line) < 2:
                    continue

                intersections = []
                for a, b in zip(source_line[:-1], source_line[1:]):
                    axy = np.asarray((a.x, a.y), dtype=float)
                    bxy = np.asarray((b.x, b.y), dtype=float)
                    for f1, f2 in trace.segments:
                        hit = segment_intersection(axy, bxy, f1, f2)
                        if hit is not None:
                            station = _project_point_to_trace(hit, trace)[2]
                            intersections.append((np.asarray(hit, float), station))

                if not intersections:
                    continue

                z = float(source_line[0].z)
                # One contour can touch/cross the same fault more than once.
                # For zero-throw evidence we need distinct contour *levels*,
                # not multiple intersections from one level.
                station = float(np.mean([item[1] for item in intersections]))
                hits_by_z.setdefault(z, []).append(station)

            crossing_levels = []
            for z, stations in hits_by_z.items():
                crossing_levels.append((z, float(np.mean(stations))))
            crossing_levels.sort(key=lambda item: item[1])
            trace.surface_crossings = tuple(crossing_levels)

            if len(crossing_levels) < min_crossings:
                continue

            stations = [item[1] for item in crossing_levels]
            if max(stations) - min(stations) < min_station_separation:
                continue

            trace.surface_barrier = False

    def slant_displacement(
        self,
        name,
        point,
        max_top_distance=None,
        fade_distance=None,
    ):
        """Return local XY top-to-bottom displacement for a slanted fault.

        The displacement is used only where the bottom trace actually supports
        the same physical top-fault piece.

        This deliberately avoids two spike-producing cases:

        1. A repeated same-name top polyline with no corresponding bottom
           polyline is left vertical.
        2. If a bottom trace terminates before the top trace, top points beyond
           that bottom support are not projected onto one bottom endpoint.
           Such endpoint collapse was the cause of the long radial spikes seen
           in the first slanted-fault implementation.

        ``max_top_distance`` is normally about 1.5 grid spacings and prevents
        an unmatched repeated top piece from borrowing another piece's bottom
        trace. ``fade_distance`` optionally tapers the tilt to zero where a
        bottom trace ends inside a longer top trace.
        """
        point = np.asarray(point, dtype=float)
        pairs = self.matched_trace_pairs(name)
        if not pairs:
            return np.zeros(2, dtype=float)

        # Select the nearest *matched* top polyline.
        best = None
        for top, bottom in pairs:
            top_hit = _project_point_to_trace(point, top)
            if best is None or top_hit[1] < best[0]:
                best = (top_hit[1], top_hit, top, bottom)

        top_distance, top_hit, top_trace, bottom_trace = best
        if max_top_distance is not None and top_distance > max_top_distance:
            return np.zeros(2, dtype=float)

        top_point, _dist, top_station, top_tangent, _at_start, _at_end = top_hit

        # Determine which interval of the top trace is actually represented by
        # this bottom trace.  This is essential when FLB is shorter than FLT.
        b0_hit = _project_point_to_trace(bottom_trace.points[0], top_trace)
        b1_hit = _project_point_to_trace(bottom_trace.points[-1], top_trace)
        support_lo = min(b0_hit[2], b1_hit[2])
        support_hi = max(b0_hit[2], b1_hit[2])

        # A tiny mismatch at coincident endpoints is normal digitizing noise.
        station_eps = 1.0e-6
        if top_station < support_lo - station_eps or top_station > support_hi + station_eps:
            return np.zeros(2, dtype=float)

        # Build the slanted plane TOP-DOWN.  The top trace is the anchor:
        # each FLB vertex is projected back onto FLT once, giving a stable
        # correspondence ``top station -> bottom XY``.  For every grid pillar
        # we first find its station on FLT and then interpolate the bottom XY
        # from that correspondence.  This follows the fault downward from the
        # structural top instead of repeatedly asking for the nearest point on
        # FLB; nearest-point selection can jump to a neighbouring bottom
        # segment at a bend and create the triangular spikes seen in F3/F4.
        bottom_point = _bottom_point_for_top_station(
            top_trace, bottom_trace, top_station
        )
        if bottom_point is None:
            return np.zeros(2, dtype=float)

        displacement = bottom_point - top_point

        # Reject only an extreme along-strike mismatch.  Normal fault dip may
        # have a modest along-strike component, so keep this guard deliberately
        # loose; its purpose is simply to prevent pathological COORD sticks.
        tangent_component = abs(float(np.dot(displacement, top_tangent)))
        normal_component = abs(float(_cross2(top_tangent, displacement)))
        if tangent_component > max(4.0 * normal_component, 4.0 * (fade_distance or 0.0), 1.0):
            return np.zeros(2, dtype=float)

        # Smoothly reduce the tilt near an *internal* FLB termination.  Do not
        # taper where top and bottom traces naturally begin/end together.
        weight = 1.0
        if fade_distance is not None and fade_distance > 0.0 and top_trace.length > _EPS:
            fade_station = min(0.25, float(fade_distance) / top_trace.length)

            if support_lo > 0.01 and top_station < support_lo + fade_station:
                weight = min(
                    weight,
                    max(0.0, (top_station - support_lo) / fade_station),
                )

            if support_hi < 0.99 and top_station > support_hi - fade_station:
                weight = min(
                    weight,
                    max(0.0, (support_hi - top_station) / fade_station),
                )

        return displacement * weight

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

        for _, c, d in self.surface_barrier_segments:
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


def _trace_cumulative_lengths(trace):
    points = np.asarray(trace.points, dtype=float)
    if len(points) < 2:
        return np.asarray([0.0]), 0.0

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return cumulative, float(cumulative[-1])


def _cross2(a, b):
    """2-D scalar cross product without NumPy's deprecated 2-vector path."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a[0] * b[1] - a[1] * b[0]


def _bottom_point_for_top_station(top_trace, bottom_trace, top_station):
    """Map a normalized station on FLT to a stable XY point on FLB.

    The bottom trace is treated as the lower interpretation of the same fault
    plane.  Every bottom vertex is projected onto the top trace, then bottom XY
    is interpolated between those projected stations.  This makes FLT the
    reference geometry and prevents nearest-segment switching at staircase
    corners.
    """
    points = np.asarray(bottom_trace.points, dtype=float)
    if len(points) == 0:
        return None
    if len(points) == 1:
        return points[0].copy()

    mapped = []
    for point in points:
        station = float(_project_point_to_trace(point, top_trace)[2])
        mapped.append((station, np.asarray(point, dtype=float)))

    mapped.sort(key=lambda item: item[0])

    # Merge virtually identical projected stations.  This can occur when FLB
    # contains a short segment almost normal to FLT.
    merged = []
    for station, point in mapped:
        if merged and abs(station - merged[-1][0]) <= 1.0e-10:
            prev_station, prev_point, count = merged[-1]
            count += 1
            merged[-1] = (
                prev_station,
                prev_point + (point - prev_point) / count,
                count,
            )
        else:
            merged.append((station, point.copy(), 1))

    if len(merged) == 1:
        return merged[0][1].copy()

    s = float(top_station)
    if s < merged[0][0] - 1.0e-10 or s > merged[-1][0] + 1.0e-10:
        return None
    if s <= merged[0][0]:
        return merged[0][1].copy()
    if s >= merged[-1][0]:
        return merged[-1][1].copy()

    stations = np.asarray([item[0] for item in merged], dtype=float)
    hi = int(np.searchsorted(stations, s, side="right"))
    lo = hi - 1
    s0, p0, _ = merged[lo]
    s1, p1, _ = merged[hi]
    span = s1 - s0
    if span <= _EPS:
        return 0.5 * (p0 + p1)
    t = (s - s0) / span
    return p0 + t * (p1 - p0)


def _point_at_trace_station(trace, station):
    """Return XY point and local tangent at normalized arclength ``station``.

    ``station`` is clamped to 0..1.  This helper is deliberately based on
    arclength rather than nearest-point geometry so matched top/bottom fault
    traces can be followed consistently from the structural top downward.
    """
    points = np.asarray(trace.points, dtype=float)
    if len(points) == 0:
        return np.zeros(2, dtype=float), np.asarray([1.0, 0.0])
    if len(points) == 1:
        return points[0].copy(), np.asarray([1.0, 0.0])

    cumulative, total_length = _trace_cumulative_lengths(trace)
    if total_length <= _EPS:
        return points[0].copy(), np.asarray([1.0, 0.0])

    station = min(1.0, max(0.0, float(station)))
    target = station * total_length
    index = int(np.searchsorted(cumulative, target, side="right") - 1)
    index = min(max(index, 0), len(points) - 2)

    a = points[index]
    b = points[index + 1]
    ab = b - a
    seg_length = float(np.linalg.norm(ab))
    if seg_length <= _EPS:
        return a.copy(), np.asarray([1.0, 0.0])

    t = (target - cumulative[index]) / seg_length
    t = min(1.0, max(0.0, float(t)))
    tangent = ab / seg_length
    return a + t * ab, tangent


def _project_point_to_trace(point, trace):
    """Project ``point`` onto a finite polyline.

    Returns:
        projected point,
        distance,
        normalized arclength station 0..1,
        local unit tangent,
        is_global_start,
        is_global_end.
    """
    point = np.asarray(point, dtype=float)
    points = np.asarray(trace.points, dtype=float)

    if len(points) == 0:
        return point.copy(), np.inf, 0.0, np.asarray([1.0, 0.0]), True, True

    if len(points) == 1:
        delta = point - points[0]
        return (
            points[0].copy(),
            float(np.linalg.norm(delta)),
            0.0,
            np.asarray([1.0, 0.0]),
            True,
            True,
        )

    cumulative, total_length = _trace_cumulative_lengths(trace)
    best = None

    for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= _EPS:
            raw_t = 0.0
        else:
            raw_t = float(np.dot(point - a, ab) / denom)

        t = min(1.0, max(0.0, raw_t))
        candidate = a + t * ab
        d2 = float(np.sum((candidate - point) ** 2))

        if best is None or d2 < best[0]:
            seg_length = float(np.linalg.norm(ab))
            tangent = ab / seg_length if seg_length > _EPS else np.asarray([1.0, 0.0])
            station_length = cumulative[index] + t * seg_length
            station = 0.0 if total_length <= _EPS else station_length / total_length
            at_start = index == 0 and t <= _EPS
            at_end = index == len(points) - 2 and t >= 1.0 - _EPS
            best = (d2, candidate, station, tangent, at_start, at_end)

    d2, candidate, station, tangent, at_start, at_end = best
    return (
        np.asarray(candidate, dtype=float),
        float(np.sqrt(d2)),
        float(station),
        np.asarray(tangent, dtype=float),
        bool(at_start),
        bool(at_end),
    )


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
    return float(_cross2(b - a, c - a))


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
    denom = _cross2(r, s)
    if abs(denom) <= eps:
        return None

    ca = c - a
    t = _cross2(ca, s) / denom
    u = _cross2(ca, r) / denom

    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        return a + t * r
    return None
