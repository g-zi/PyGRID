"""Reading and interpolation of TinyECL map data."""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree, ConvexHull


@dataclass
class MapPoint:
    x: float
    y: float
    z: float
    id: str
    # Internal contiguous-polyline identifier.  TinyECL's visible line counter
    # is not guaranteed to be globally unique, so interpolation must not use
    # the text id alone to decide which samples belong to one contour line.
    line_id: int = 0


class ScatteredMap:
    """Base class for TinyECL XYZ-style scattered map data.

    TinyECL writes contour vertices.  A contour is a *line* interpretation,
    however a scattered interpolator only sees points.  PyGRID therefore keeps
    each contiguous contour run and can densify it to a regular point spacing
    before surface interpolation.  This mirrors the workflow where contour
    lines are converted to many XYZ samples instead of being used directly as
    mathematical surface boundaries.
    """

    def __init__(self, filename):
        self.filename = Path(filename)
        self.points = []
        self._source_lines = []
        self._linear = None
        self._nearest = None
        self._xy = None
        self._z = None
        self._tree = None
        self._line_groups = None
        self._level_groups = None
        self._level_trees = None
        self._hull_segments = None
        # GridModel sets this to several grid-block widths.  Points inside this
        # band are treated as edge extrapolation as well, because a convex-hull
        # Delaunay triangle can still bridge unsupported contour endpoints.
        self.edge_extrapolation_distance = 0.0
        self.sample_spacing = None
        self.source_point_count = 0

    def read(self):
        self.points = []
        self._source_lines = []

        current_key = None
        current_line = None
        line_id = -1

        with open(self.filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("*") or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                identifier = parts[3] if len(parts) >= 4 else ""

                # A TinyECL contour is written as one contiguous run.  Start a
                # new internal line whenever either its source identifier or Z
                # changes.  This is intentionally stricter than grouping only
                # by identifier because old workbooks can reuse a line counter.
                key = (identifier, z)
                if current_line is None or key != current_key:
                    line_id += 1
                    current_line = []
                    self._source_lines.append(current_line)
                    current_key = key

                point = MapPoint(x, y, z, identifier, line_id)
                current_line.append(point)
                self.points.append(point)

        if len(self.points) < 3:
            raise ValueError(
                f"Map '{self.filename}' must contain at least 3 XYZ points"
            )

        self.source_point_count = len(self.points)
        self.sample_spacing = None
        self._build_interpolators()
        return self

    def densify(self, spacing):
        """Resample each contiguous contour into regularly spaced XYZ points.

        Original vertices are retained exactly.  Additional points are inserted
        along every digitized segment so no segment between neighbouring source
        vertices is longer than ``spacing``.  Z is linearly interpolated along
        the segment (for a true contour it is normally constant).
        """
        spacing = float(spacing)
        if spacing <= 0.0:
            raise ValueError("Contour sample spacing must be > 0")

        if not self._source_lines:
            # Support callers that constructed points programmatically.
            self._source_lines = [[p] for p in self.points]

        dense = []

        for line_id, source in enumerate(self._source_lines):
            if not source:
                continue

            if len(source) == 1:
                p = source[0]
                dense.append(MapPoint(p.x, p.y, p.z, p.id, line_id))
                continue

            # Keep the first source vertex exactly.
            p0 = source[0]
            dense.append(MapPoint(p0.x, p0.y, p0.z, p0.id, line_id))

            for p1 in source[1:]:
                dx = p1.x - p0.x
                dy = p1.y - p0.y
                distance = math.hypot(dx, dy)
                steps = max(1, int(math.ceil(distance / spacing)))

                for step in range(1, steps + 1):
                    t = step / steps
                    dense.append(
                        MapPoint(
                            p0.x + t * dx,
                            p0.y + t * dy,
                            p0.z + t * (p1.z - p0.z),
                            p1.id,
                            line_id,
                        )
                    )

                p0 = p1

        self.points = dense
        self.sample_spacing = spacing
        self._build_interpolators()
        return self

    def _build_interpolators(self):
        self._xy = np.asarray([(p.x, p.y) for p in self.points], dtype=float)
        self._z = np.asarray([p.z for p in self.points], dtype=float)

        self._linear = LinearNDInterpolator(self._xy, self._z, fill_value=np.nan)
        self._nearest = NearestNDInterpolator(self._xy, self._z)
        self._tree = cKDTree(self._xy)

        groups = {}
        for idx, point in enumerate(self.points):
            groups.setdefault(point.line_id, []).append(idx)
        self._line_groups = groups

        # Distinct contour levels are also kept explicitly for edge
        # extrapolation.  Multiple separate polylines with the same Z value
        # belong to one structural level and must not masquerade as independent
        # depth information in a local trend fit.
        level_groups = {}
        for idx, point in enumerate(self.points):
            level_groups.setdefault(float(point.z), []).append(idx)
        self._level_groups = {
            level: np.asarray(indices, dtype=int)
            for level, indices in level_groups.items()
        }
        self._level_trees = {
            level: cKDTree(self._xy[indices])
            for level, indices in self._level_groups.items()
        }

        hull = ConvexHull(self._xy)
        hull_points = self._xy[hull.vertices]
        self._hull_segments = [
            (hull_points[i], hull_points[(i + 1) % len(hull_points)])
            for i in range(len(hull_points))
        ]

    def _edge_extrapolation_mask(self, queries):
        """Return queries outside or close to the data convex-hull edge.

        A query can be technically inside the global convex hull while the
        Delaunay triangulation spans a large unsupported gap between contour
        endpoints.  Near the map boundary that behaves like extrapolation and
        is exactly where the old edge-collapse artefact occurred.
        """
        q2 = np.atleast_2d(np.asarray(queries, dtype=float))
        probe = np.asarray(self._linear(q2), dtype=float).reshape(-1)
        mask = np.isnan(probe)

        width = float(getattr(self, "edge_extrapolation_distance", 0.0) or 0.0)
        if width <= 0.0 or not self._hull_segments:
            return mask

        inside = np.flatnonzero(~mask)
        if len(inside) == 0:
            return mask

        qi = q2[inside]
        min_d2 = np.full(len(qi), np.inf, dtype=float)
        for a, b in self._hull_segments:
            v = b - a
            vv = float(np.dot(v, v))
            if vv <= 1.0e-24:
                d2 = np.sum((qi - a) ** 2, axis=1)
            else:
                t = ((qi - a) @ v) / vv
                t = np.clip(t, 0.0, 1.0)
                projection = a + t[:, None] * v
                d2 = np.sum((qi - projection) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

        mask[inside] = min_d2 <= width * width
        return mask

    def _local_planar_extrapolate(
        self,
        queries,
        faults=None,
        barrier_origins=None,
        max_levels=8,
        points_per_level=3,
    ):
        """Extrapolate outside the contour hull from a local structural trend.

        The previous PyGRID edge fallback used the nearest contour value (or,
        in faulted mode, local inverse-distance averaging).  That can flatten a
        surface at the map boundary and may even make BOTTOM cross TOP.

        For each outside-hull query this routine instead finds nearby samples
        from *different contour Z levels*, rejects samples across a hard fault
        barrier, and fits a weighted local plane::

            Z = a * (X-Xq) + b * (Y-Yq) + c

        At least two distinct contour levels and a full-rank XY fit are required.
        Three or more levels are preferred automatically.  If the local geometry
        cannot define a stable plane, the routine falls back to the nearest
        reachable sample rather than inventing a gradient.
        """
        if self._xy is None or self._level_groups is None:
            self._build_interpolators()

        q2 = np.atleast_2d(np.asarray(queries, dtype=float))
        if barrier_origins is None:
            origins = q2
        else:
            origins = np.atleast_2d(np.asarray(barrier_origins, dtype=float))
            if len(origins) != len(q2):
                raise ValueError(
                    "barrier_origins must have one XY point per query"
                )

        out = np.empty(len(q2), dtype=float)
        local_scale = float(self.sample_spacing or 0.0)
        if local_scale <= 1.0e-12:
            # A robust scale keeps the WLS weights well behaved if densification
            # was disabled in a hand-written PyGRID run file.
            if len(self._xy) > 1:
                d, _ = self._tree.query(self._xy, k=2)
                d = np.asarray(d, dtype=float)[:, 1]
                d = d[np.isfinite(d) & (d > 1.0e-12)]
                local_scale = float(np.median(d)) if len(d) else 1.0
            else:
                local_scale = 1.0

        level_items = list(self._level_groups.items())

        # Batch all KD-tree lookups by contour level.  cKDTree handles an
        # array of queries efficiently; doing this once per level is much
        # faster than issuing the same lookup separately for every edge point.
        raw_by_level = {}
        nearest_by_level = {}
        for level, global_indices in level_items:
            tree = self._level_trees[level]
            k = min(max(points_per_level * 2, 6), len(global_indices))
            distances, local_indices = tree.query(q2, k=k)
            distances = np.asarray(distances, dtype=float)
            local_indices = np.asarray(local_indices, dtype=int)
            if k == 1:
                distances = distances.reshape((len(q2), 1))
                local_indices = local_indices.reshape((len(q2), 1))
            elif distances.ndim == 1:
                distances = distances.reshape((1, -1))
                local_indices = local_indices.reshape((1, -1))
            global_pick = global_indices[local_indices]
            raw_by_level[level] = (global_pick, distances)
            nearest_by_level[level] = distances[:, 0]

        for qi, q in enumerate(q2):
            origin = origins[qi]

            ranked_levels = sorted(
                (float(nearest_by_level[level][qi]), level)
                for level, _global_indices in level_items
            )
            ranked_levels = ranked_levels[
                :min(len(ranked_levels), max_levels + 4)
            ]

            if faults and ranked_levels:
                flat_indices = []
                slices = {}
                offset = 0
                for _nearest, level in ranked_levels:
                    indices = raw_by_level[level][0][qi]
                    flat_indices.extend(indices.tolist())
                    slices[level] = (offset, offset + len(indices))
                    offset += len(indices)
                flat_indices = np.asarray(flat_indices, dtype=int)
                flat_blocked = faults.blocked_mask(
                    origin, self._xy[flat_indices]
                )
            else:
                flat_blocked = None
                slices = {}

            level_candidates = []
            for nearest_distance, level in ranked_levels:
                global_pick = raw_by_level[level][0][qi]
                distances = raw_by_level[level][1][qi]
                if flat_blocked is not None:
                    lo, hi = slices[level]
                    keep = np.flatnonzero(~flat_blocked[lo:hi])
                else:
                    keep = np.arange(len(global_pick), dtype=int)
                if len(keep) == 0:
                    continue
                keep = keep[:points_per_level]
                chosen = global_pick[keep]
                chosen_dist = distances[keep]
                level_candidates.append(
                    (float(np.min(chosen_dist)), level, chosen, chosen_dist)
                )
                if len(level_candidates) >= max_levels:
                    break

            if not level_candidates:
                out[qi] = float(np.asarray(self._nearest(q)).reshape(-1)[0])
                continue

            level_candidates.sort(key=lambda item: item[0])

            def longest_monotonic(items, increasing):
                n = len(items)
                if n == 0:
                    return []
                length = [1] * n
                previous = [-1] * n
                for ii in range(n):
                    zi = float(items[ii][1])
                    for jj in range(ii):
                        zj = float(items[jj][1])
                        if increasing:
                            ordered = zj < zi - 1.0e-9
                        else:
                            ordered = zj > zi + 1.0e-9
                        if ordered and length[jj] + 1 > length[ii]:
                            length[ii] = length[jj] + 1
                            previous[ii] = jj
                end_index = max(range(n), key=lambda k: length[k])
                result_items = []
                while end_index >= 0:
                    result_items.append(items[end_index])
                    end_index = previous[end_index]
                result_items.reverse()
                return result_items

            increasing = longest_monotonic(level_candidates, True)
            decreasing = longest_monotonic(level_candidates, False)

            if len(increasing) > len(decreasing):
                coherent = increasing
            elif len(decreasing) > len(increasing):
                coherent = decreasing
            else:
                inc_distance = (
                    float(np.mean([item[0] for item in increasing]))
                    if increasing else np.inf
                )
                dec_distance = (
                    float(np.mean([item[0] for item in decreasing]))
                    if decreasing else np.inf
                )
                coherent = increasing if inc_distance < dec_distance else decreasing

            if len(coherent) < 3:
                coherent = level_candidates

            fitted = False
            if len(coherent) >= 2:
                matrices = []
                values_local = []
                weights_local = []
                for _nearest_distance, _level, indices, dists in coherent:
                    c = self._xy[indices]
                    d = self._z[indices]
                    delta_local = c - q
                    A = np.column_stack(
                        (
                            delta_local[:, 0],
                            delta_local[:, 1],
                            np.ones(len(c)),
                        )
                    )
                    w = 1.0 / np.square(dists + local_scale)
                    total = float(np.sum(w))
                    if total <= 0.0 or not np.isfinite(total):
                        continue
                    w = w / total
                    matrices.append(A)
                    values_local.append(d)
                    weights_local.append(w)

                if matrices:
                    A = np.vstack(matrices)
                    d = np.concatenate(values_local)
                    w = np.concatenate(weights_local)
                    root_w = np.sqrt(w)
                    coeff, _residuals, rank, singular = np.linalg.lstsq(
                        A * root_w[:, None], d * root_w, rcond=None
                    )
                    if (
                        rank == 3
                        and np.all(np.isfinite(coeff))
                        and len(singular) == 3
                        and singular[-1] > 0.0
                        and singular[0] / singular[-1] < 1.0e10
                    ):
                        out[qi] = float(coeff[2])
                        fitted = True

            if not fitted:
                nearest = min(level_candidates, key=lambda item: item[0])
                out[qi] = float(self._z[nearest[2][0]])

        return out

    def interpolate(self, xy):
        """Ordinary unfaulted interpolation."""
        if self._linear is None or self._nearest is None:
            self._build_interpolators()

        query = np.asarray(xy, dtype=float)
        single = query.ndim == 1
        query2 = np.atleast_2d(query)

        values = np.asarray(self._linear(query2), dtype=float).reshape(-1)
        edge = self._edge_extrapolation_mask(query2)
        if np.any(edge):
            values[edge] = self._local_planar_extrapolate(query2[edge])

        return float(values[0]) if single else values

    def interpolate_faulted(
        self,
        xy,
        faults,
        search_radius=None,
        per_line=3,
        max_neighbours=8,
        barrier_origins=None,
    ):
        """Interpolate while treating every fault trace as a hard barrier.

        The contour input is already represented by regularly sampled XYZ
        points when GridModel contour densification is enabled.  Thus each side
        of a fault can use samples close to the actual contour/fault crossing,
        instead of extrapolating from a few distant hand-digitized vertices.

        ``search_radius`` is the GRID-style search radius for datapoints, in
        model coordinate units.  A larger radius allows more interpreted data
        to contribute and generally gives a smoother surface.  No datapoint
        across a fault may contribute, irrespective of radius.

        ``None`` preserves the original PyGRID v1 nearest-line behaviour.
        """
        if not faults:
            return self.interpolate(xy)

        if self._xy is None:
            self._build_interpolators()

        if search_radius is not None:
            search_radius = float(search_radius)
            if search_radius <= 0.0:
                raise ValueError("model.search_radius must be > 0 or None")

        query = np.asarray(xy, dtype=float)
        single = query.ndim == 1
        queries = np.atleast_2d(query)
        result = np.empty(len(queries), dtype=float)

        if barrier_origins is None:
            barrier_origins2 = queries
        else:
            barrier_origins2 = np.atleast_2d(
                np.asarray(barrier_origins, dtype=float)
            )
            if len(barrier_origins2) != len(queries):
                raise ValueError(
                    "barrier_origins must have one XY point per query"
                )

        group_arrays = []
        if search_radius is None:
            # Only needed for the legacy no-radius mode.
            for indices in self._line_groups.values():
                idx = np.asarray(indices, dtype=int)
                group_arrays.append((idx, self._xy[idx]))

        # Near the global data boundary, a Delaunay triangle may bridge an
        # unsupported gap between contour endpoints even though the query is
        # technically inside the convex hull.  Treat that edge band as
        # extrapolation too; the returned Z remains fault-barrier aware.
        edge_extrapolation = self._edge_extrapolation_mask(queries)
        edge_indices = np.flatnonzero(edge_extrapolation)
        if len(edge_indices):
            result[edge_indices] = self._local_planar_extrapolate(
                queries[edge_indices],
                faults=faults,
                barrier_origins=barrier_origins2[edge_indices],
            )

        for qi, q in enumerate(queries):
            barrier_origin = barrier_origins2[qi]

            if edge_extrapolation[qi]:
                continue

            if search_radius is not None:
                # KD-tree radius lookup avoids scanning every digitized point
                # for every grid corner.  This is both faster and exactly what
                # the search-radius concept requires.
                candidate_indices = np.asarray(
                    self._tree.query_ball_point(q, r=search_radius), dtype=int
                )
                if len(candidate_indices):
                    delta = self._xy[candidate_indices] - q
                    candidate_distances = np.sqrt(np.sum(delta * delta, axis=1))
                else:
                    candidate_distances = np.empty(0, dtype=float)
            else:
                candidate_indices_list = []
                candidate_distances_list = []
                for indices, coords in group_arrays:
                    d2 = np.sum((coords - q) ** 2, axis=1)
                    take = min(per_line, len(indices))
                    local = (
                        np.argpartition(d2, take - 1)[:take]
                        if take < len(indices)
                        else np.arange(len(indices))
                    )
                    candidate_indices_list.extend(indices[local].tolist())
                    candidate_distances_list.extend(np.sqrt(d2[local]).tolist())
                candidate_indices = np.asarray(candidate_indices_list, dtype=int)
                candidate_distances = np.asarray(candidate_distances_list, dtype=float)

            exact = candidate_distances <= 1.0e-10
            if np.any(exact):
                result[qi] = self._z[candidate_indices[np.flatnonzero(exact)[0]]]
                continue

            if len(candidate_indices):
                # Check nearest candidates first.  Limiting the barrier test
                # keeps large search radii practical while still allowing far
                # more points than the old eight-neighbour interpolation.
                order = np.argsort(candidate_distances)
                if search_radius is not None:
                    order = order[:64]
                candidate_indices = candidate_indices[order]
                candidate_distances = candidate_distances[order]

                blocked = faults.blocked_mask(
                    barrier_origin, self._xy[candidate_indices]
                )
                keep = ~blocked
                candidate_indices = candidate_indices[keep]
                candidate_distances = candidate_distances[keep]

            if len(candidate_indices) == 0:
                # Radius empty, or every point inside it lies across a fault:
                # use the nearest reachable sample rather than crossing a fault.
                k = min(128, len(self._xy))
                distances, indices = self._tree.query(q, k=k)
                distances = np.atleast_1d(distances)
                indices = np.atleast_1d(indices).astype(int)
                blocked = faults.blocked_mask(
                    barrier_origin, self._xy[indices]
                )
                usable = np.flatnonzero(~blocked)
                if len(usable):
                    pos = usable[0]
                    candidate_indices = np.asarray([indices[pos]])
                    candidate_distances = np.asarray([distances[pos]])

            if len(candidate_indices) == 0:
                result[qi] = float(np.asarray(self._nearest(q)).reshape(-1)[0])
                continue

            if search_radius is None:
                order = np.argsort(candidate_distances)[:max_neighbours]
                candidate_indices = candidate_indices[order]
                candidate_distances = candidate_distances[order]

            # Inverse-distance interpolation.  With a search radius, every
            # reachable candidate retained above contributes, so increasing the
            # radius progressively reduces the imprint of individual points.
            weights = 1.0 / np.maximum(candidate_distances, 1.0e-12)
            result[qi] = float(
                np.sum(weights * self._z[candidate_indices]) / np.sum(weights)
            )

        return float(result[0]) if single else result


class ContourMap(ScatteredMap):
    pass


class ThicknessMap(ScatteredMap):
    pass
