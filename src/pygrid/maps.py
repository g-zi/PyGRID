"""Reading and interpolation of TinyECL map data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree


@dataclass
class MapPoint:
    x: float
    y: float
    z: float
    id: str


class ScatteredMap:
    """Base class for TinyECL XYZ-style scattered map data."""

    def __init__(self, filename):
        self.filename = Path(filename)
        self.points = []
        self._linear = None
        self._nearest = None
        self._xy = None
        self._z = None
        self._tree = None
        self._line_groups = None

    def read(self):
        self.points = []

        with open(self.filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("*") or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                identifier = parts[3] if len(parts) >= 4 else ""
                self.points.append(
                    MapPoint(
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        identifier,
                    )
                )

        if len(self.points) < 3:
            raise ValueError(
                f"Map '{self.filename}' must contain at least 3 XYZ points"
            )

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
            groups.setdefault(point.id, []).append(idx)
        self._line_groups = groups

    def interpolate(self, xy):
        """Ordinary unfaulted interpolation."""
        if self._linear is None or self._nearest is None:
            self._build_interpolators()

        query = np.asarray(xy, dtype=float)
        single = query.ndim == 1
        query2 = np.atleast_2d(query)

        values = np.asarray(self._linear(query2), dtype=float).reshape(-1)
        missing = np.isnan(values)
        if np.any(missing):
            values[missing] = np.asarray(
                self._nearest(query2[missing]), dtype=float
            ).reshape(-1)

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

        for qi, q in enumerate(queries):
            barrier_origin = barrier_origins2[qi]
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
                blocked = faults.blocked_mask(barrier_origin, self._xy[indices])
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
