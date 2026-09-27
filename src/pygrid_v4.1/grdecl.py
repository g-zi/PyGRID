"""Eclipse GRDECL output."""

import numpy as np
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .faults import (FaultSet, FaultTrace, segment_intersection,
    _project_point_to_trace, _point_at_trace_station,
    _bottom_point_for_top_station)
from .geometry import create_pillars, read_extent
from .version import PYGRID_VERSION, PYGRID_BUILD, FAULT_GEOMETRY_REVISION


class GRDECLWriter:

    PILLAR_Z_TOP = 0.0
    PILLAR_Z_BOTTOM = 100000.0
    FAULT_SIDE_NUDGE = 1.0e-3

    def __init__(self, model):
        self.model = model
        self._fault_faces_cache = None
        self._surface_fault_face_sets_cache = {}
        self._surface_fault_sets_cache = {}
        self._interfaces_cache = None

    @staticmethod
    def _normalized_unit(unit):
        """Return the Eclipse spelling for a supported length unit."""
        value = str(unit or "").strip().upper()
        aliases = {
            "M": "METRES",
            "METRE": "METRES",
            "METRES": "METRES",
            "METER": "METRES",
            "METERS": "METRES",
            "FT": "FEET",
            "FOOT": "FEET",
            "FEET": "FEET",
        }
        if value not in aliases:
            raise ValueError(
                f"Unsupported PyGRID length unit {unit!r}; use METRES or FEET"
            )
        return aliases[value]

    @classmethod
    def _unit_factor(cls, from_unit, to_unit):
        """Multiplicative factor converting one supported length unit."""
        source = cls._normalized_unit(from_unit)
        target = cls._normalized_unit(to_unit)
        if source == target:
            return 1.0
        if source == "FEET" and target == "METRES":
            return 0.3048
        if source == "METRES" and target == "FEET":
            return 1.0 / 0.3048
        raise ValueError(f"Cannot convert {source} to {target}")

    def _map_units(self):
        return self._normalized_unit(
            getattr(self.model, "surface_units", self.model.units)
        )

    def _grid_units(self):
        # TinyECL convention: GRIDUNIT follows the reservoir/depth system.
        return self._normalized_unit(
            getattr(self.model, "depth_units", self.model.units)
        )

    def _mapaxes(self):
        """Return Eclipse MAPAXES in map/surface coordinate units.

        PyGRID input XY values are already in the global map coordinate system.
        GRDECL output is deliberately written as a *local* grid in GRIDUNIT so
        that X, Y and Z all use one consistent unit (FEET for FIELD projects,
        METRES for metric projects).  MAPAXES then places that local grid back
        into the original map system.

        The default axes are derived from the TinyECL extent rectangle:
        positive grid X follows NW->NE and positive grid Y follows NW->SW.
        The two MAPAXES vectors are written with equal length, as expected by
        Eclipse-family readers.  An explicit model.mapaxes = (x1,y1,x0,y0,x2,y2)
        may be supplied later if a project wants to preserve an external map
        frame exactly.
        """
        explicit = getattr(self.model, "mapaxes", None)
        if explicit is not None:
            values = np.asarray(explicit, dtype=float).reshape(-1)
            if len(values) != 6 or not np.all(np.isfinite(values)):
                raise ValueError(
                    "model.mapaxes must contain six finite values: "
                    "x1, y1, x0, y0, x2, y2"
                )
            x1, y1, x0, y0, x2, y2 = values
            xvec = np.asarray([x2 - x0, y2 - y0], dtype=float)
            yvec = np.asarray([x1 - x0, y1 - y0], dtype=float)
            if np.linalg.norm(xvec) <= 1.0e-12 or np.linalg.norm(yvec) <= 1.0e-12:
                raise ValueError("model.mapaxes contains a zero-length axis")
            return tuple(float(v) for v in values)

        if not self.model.extent:
            raise ValueError("model.extent must be set before writing MAPAXES")

        nw, sw, _se, ne = np.asarray(read_extent(self.model.extent), dtype=float)
        xvec = np.asarray(ne - nw, dtype=float)
        yvec = np.asarray(sw - nw, dtype=float)
        xlen = float(np.linalg.norm(xvec))
        ylen = float(np.linalg.norm(yvec))
        if xlen <= 1.0e-12 or ylen <= 1.0e-12:
            raise ValueError("Grid extent contains a zero-length X or Y edge")

        xhat = xvec / xlen
        yhat = yvec / ylen
        axis_length = max(xlen, ylen, 1.0)
        origin = nw
        p_y = origin + yhat * axis_length
        p_x = origin + xhat * axis_length
        return (
            float(p_y[0]), float(p_y[1]),
            float(origin[0]), float(origin[1]),
            float(p_x[0]), float(p_x[1]),
        )

    def _map_basis(self):
        """Return map origin and unit X/Y basis vectors from MAPAXES."""
        x1, y1, x0, y0, x2, y2 = self._mapaxes()
        origin = np.asarray([x0, y0], dtype=float)
        xvec = np.asarray([x2 - x0, y2 - y0], dtype=float)
        yvec = np.asarray([x1 - x0, y1 - y0], dtype=float)
        xhat = xvec / np.linalg.norm(xvec)
        yhat = yvec / np.linalg.norm(yvec)
        basis = np.column_stack((xhat, yhat))
        if abs(float(np.linalg.det(basis))) <= 1.0e-12:
            raise ValueError("MAPAXES X and Y axes are collinear")
        return origin, basis

    def _map_xy_to_grid_xy(self, xy):
        """Transform global map XY to local GRIDUNIT coordinates."""
        points = np.asarray(xy, dtype=float)
        original_shape = points.shape
        points2 = points.reshape((-1, 2))
        origin, basis = self._map_basis()
        local_surface = np.linalg.solve(
            basis, (points2 - origin).T
        ).T
        factor = self._unit_factor(self._map_units(), self._grid_units())
        return (local_surface * factor).reshape(original_shape)

    def write_map_header(self, f):
        """Write MAPUNITS, MAPAXES and GRIDUNIT before SPECGRID."""
        map_units = self._map_units()
        grid_units = self._grid_units()
        x1, y1, x0, y0, x2, y2 = self._mapaxes()

        f.write("MAPUNITS\n")
        f.write("-- Units for map coordinates\n")
        f.write(f"  {map_units} /\n\n")

        f.write("MAPAXES\n")
        f.write("-- Grid axes with respect to map coordinates\n")
        f.write(f"  {x1:.6f} {y1:.6f}\n")
        f.write(f"  {x0:.6f} {y0:.6f}\n")
        f.write(f"  {x2:.6f} {y2:.6f} /\n\n")

        f.write("GRIDUNIT\n")
        f.write("-- Units for grid coordinates\n")
        f.write(f"  {grid_units} /\n\n")

    def write_specgrid(self, f):
        if self.model.layers is None:
            raise ValueError("model.layers must be set before writing SPECGRID")

        f.write("SPECGRID\n")
        f.write(
            f"  {self.model.nx} {self.model.ny} "
            f"{self.model.layers} 1 F /\n\n"
        )

    def _pillars(self):
        if not self.model.extent:
            raise ValueError("model.extent must be set before writing grid geometry")
        corners = read_extent(self.model.extent)
        return create_pillars(corners, self.model.nx, self.model.ny)

    def _base_pillar_array(self):
        """Return the undeformed logical XY pillar grid."""
        return np.asarray(self._pillars(), dtype=float).reshape(
            (self.model.ny + 1, self.model.nx + 1, 2)
        )

    def _conforming_fault_names(self):
        """Return the TOP fault names that should use conforming geometry."""
        if (
            not bool(getattr(self.model, "conform_slanted_faults", False))
            or not self.model.split_faults
            or not self.model.fault_set
        ):
            return set()

        mode = self._slanted_fault_mode()
        if mode == "COORD":
            # ZigZag OFF: every TOP fault conforms.
            return set(self.model.fault_set.names)
        if mode == "MIXED":
            # ZigZag ON: only a same-name exported FLB opts that fault into
            # conforming/slanted geometry.  TOP-only faults stay zigzag.
            return set(self.model.fault_set.slanted_names)
        return set()

    def _pillar_array(self):
        """Return TOP XY pillars with the selected per-fault conformity."""
        names = self._conforming_fault_names()
        if names:
            return self._conformed_top_pillar_xy(names=names)
        return self._base_pillar_array()

    def _faces_for_fault_set_on_centres(self, fault_set, centres):
        """Convert fault traces to logical faces using explicitly supplied centres.

        This small helper is important for fault-conforming grids: the discrete
        topology is chosen once on the original undeformed grid, then the same
        row/column path is retained while its XY pillars are moved onto FLT.
        """
        if not fault_set:
            return []

        centres = np.asarray(centres, dtype=float)
        nx, ny = self.model.nx, self.model.ny
        found = set()
        for trace in fault_set.traces:
            for f1, f2 in trace.segments:
                for j in range(ny):
                    for i in range(nx - 1):
                        if segment_intersection(
                            centres[j, i], centres[j, i + 1], f1, f2
                        ) is not None:
                            found.add((trace.name, i + 1, j + 1, "X+"))
                for j in range(ny - 1):
                    for i in range(nx):
                        if segment_intersection(
                            centres[j, i], centres[j + 1, i], f1, f2
                        ) is not None:
                            found.add((trace.name, i + 1, j + 1, "Y+"))
        return sorted(found, key=lambda x: (x[0], x[2], x[1], x[3]))

    @staticmethod
    def _ordered_pillar_components_from_faces(faces, name):
        """Return ordered pillar chains for ``name`` from explicit face records."""
        adjacency = {}
        for face_name, i1, j1, face in faces:
            if face_name != name:
                continue
            endpoints = GRDECLWriter._fault_edge_endpoints(i1, j1, face)
            if len(endpoints) != 2:
                continue
            a, b = endpoints
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        components = []
        unseen = set(adjacency)
        while unseen:
            seed = next(iter(unseen))
            stack = [seed]
            comp = set()
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                unseen.discard(node)
                stack.extend(adjacency.get(node, ()))

            ends = sorted(
                node for node in comp if len(adjacency.get(node, ())) == 1
            )
            start = ends[0] if ends else min(comp)
            ordered = [start]
            previous = None
            current = start
            while True:
                candidates = [
                    node for node in adjacency.get(current, ())
                    if node != previous and node in comp
                ]
                if not candidates:
                    break
                unvisited = [node for node in candidates if node not in ordered]
                if not unvisited:
                    break
                nxt = sorted(unvisited)[0]
                ordered.append(nxt)
                previous, current = current, nxt
                if len(ordered) > len(comp) + 2:
                    break
            components.append(ordered)
        return components

    def _legacy_raw_fault_faces(self):
        """Return the proven v3.22 centre-crossing logical fault path."""
        base = self._base_pillar_array()
        centres = self._cell_centres_from_pillars(base)
        return self._faces_for_fault_set_on_centres(
            self._fault_set_for_level("top"), centres
        )

    def _representative_xy_edge_spacing(self, base):
        lengths = []
        for j in range(self.model.ny + 1):
            for i in range(self.model.nx):
                lengths.append(float(np.linalg.norm(base[j, i + 1] - base[j, i])))
        for j in range(self.model.ny):
            for i in range(self.model.nx + 1):
                lengths.append(float(np.linalg.norm(base[j + 1, i] - base[j, i])))
        finite = np.asarray([v for v in lengths if np.isfinite(v) and v > 1.0e-12])
        return float(np.median(finite)) if finite.size else 1.0

    def _fault_path_edge_cost(
        self, base, trace, u, v, step_index, step_count,
        start_station, end_station, previous_axis=None,
    ):
        """V4 cost for one candidate logical fault edge.

        The cost balances proximity to FLT, alignment with its local tangent,
        along-trace station progress and excessive I/J switching.  All terms
        are dimensionless so the same defaults work for metric and FIELD maps.
        """
        midpoint = 0.5 * (np.asarray(base[u], float) + np.asarray(base[v], float))
        _hit, distance, station, tangent, _at_start, _at_end = (
            _project_point_to_trace(midpoint, trace)
        )
        edge = np.asarray(base[v], float) - np.asarray(base[u], float)
        edge_len = float(np.linalg.norm(edge))
        tangent_len = float(np.linalg.norm(tangent))
        if edge_len <= 1.0e-12 or tangent_len <= 1.0e-12:
            return np.inf
        edge /= edge_len
        tangent = np.asarray(tangent, float) / tangent_len
        spacing = self._representative_xy_edge_spacing(base)

        distance_term = (float(distance) / max(spacing, 1.0e-12)) ** 2
        alignment_term = 1.0 - abs(float(np.dot(edge, tangent)))
        expected_station = float(start_station) + (
            (float(step_index) + 0.5) / max(float(step_count), 1.0)
        ) * (float(end_station) - float(start_station))
        station_span = max(abs(float(end_station) - float(start_station)), 1.0e-9)
        station_term = ((float(station) - expected_station) / station_span) ** 2

        axis = "I" if u[0] == v[0] else "J"
        turn_term = 1.0 if previous_axis is not None and axis != previous_axis else 0.0
        return (
            float(getattr(self.model, "fault_path_distance_weight", 1.0)) * distance_term
            + float(getattr(self.model, "fault_path_alignment_weight", 1.0)) * alignment_term
            + float(getattr(self.model, "fault_path_station_weight", 1.0)) * station_term
            + float(getattr(self.model, "fault_path_turn_weight", 0.10)) * turn_term
        )

    def _fault_path_objective(self, base, trace, path):
        if len(path) < 2:
            return np.inf
        _p0, _d0, s0, _t0, _a0, _b0 = _project_point_to_trace(base[path[0]], trace)
        _p1, _d1, s1, _t1, _a1, _b1 = _project_point_to_trace(base[path[-1]], trace)
        total = 0.0
        previous_axis = None
        step_count = len(path) - 1
        for step, (u, v) in enumerate(zip(path[:-1], path[1:])):
            total += self._fault_path_edge_cost(
                base, trace, u, v, step, step_count, s0, s1, previous_axis
            )
            previous_axis = "I" if u[0] == v[0] else "J"
        return float(total)

    def _optimized_monotonic_fault_path(self, base, trace, legacy_chain):
        """Return a lower-cost monotonic edge path with v3.22 endpoints.

        Non-monotonic, branched or otherwise unusual legacy components are not
        changed.  This is intentional: V4 is an experimental improvement path,
        not a reason to weaken the robust v3.22 fallback.
        """
        if len(legacy_chain) < 2:
            return list(legacy_chain), np.inf, np.inf
        start = tuple(legacy_chain[0])
        end = tuple(legacy_chain[-1])
        dj_total = int(end[0] - start[0])
        di_total = int(end[1] - start[1])
        step_count = abs(dj_total) + abs(di_total)
        if step_count != len(legacy_chain) - 1:
            cost = self._fault_path_objective(base, trace, legacy_chain)
            return list(legacy_chain), cost, cost

        dj = 0 if dj_total == 0 else (1 if dj_total > 0 else -1)
        di = 0 if di_total == 0 else (1 if di_total > 0 else -1)
        _p0, _d0, s0, _t0, _a0, _b0 = _project_point_to_trace(base[start], trace)
        _p1, _d1, s1, _t1, _a1, _b1 = _project_point_to_trace(base[end], trace)

        states = {(start[0], start[1], None): (0.0, [start])}
        nx, ny = self.model.nx, self.model.ny
        for step in range(step_count):
            next_states = {}
            for (j, i, previous_axis), (cost, path) in states.items():
                options = []
                if j != end[0]:
                    options.append(("J", (j + dj, i)))
                if i != end[1]:
                    options.append(("I", (j, i + di)))

                for axis, v in options:
                    jj, ii = v
                    if not (0 <= jj <= ny and 0 <= ii <= nx):
                        continue
                    # A fault edge must separate two cells.  Walking *along*
                    # the outside model boundary would create no Eclipse face.
                    if axis == "J" and not (0 < i < nx):
                        continue
                    if axis == "I" and not (0 < j < ny):
                        continue
                    edge_cost = self._fault_path_edge_cost(
                        base, trace, (j, i), v, step, step_count,
                        s0, s1, previous_axis,
                    )
                    if not np.isfinite(edge_cost):
                        continue
                    key = (jj, ii, axis)
                    candidate = (cost + edge_cost, path + [v])
                    if key not in next_states or candidate[0] < next_states[key][0]:
                        next_states[key] = candidate
            states = next_states
            if not states:
                legacy_cost = self._fault_path_objective(base, trace, legacy_chain)
                return list(legacy_chain), legacy_cost, legacy_cost

        candidates = [
            item for key, item in states.items() if key[:2] == end
        ]
        legacy_cost = self._fault_path_objective(base, trace, legacy_chain)
        if not candidates:
            return list(legacy_chain), legacy_cost, legacy_cost
        candidate_cost, candidate_path = min(candidates, key=lambda item: item[0])
        min_improvement = float(
            getattr(self.model, "fault_path_min_improvement", 1.0e-9)
        )
        if candidate_cost < legacy_cost - min_improvement:
            return candidate_path, legacy_cost, float(candidate_cost)
        return list(legacy_chain), legacy_cost, legacy_cost

    def _faces_from_fault_pillar_path(self, name, path):
        faces = []
        nx, ny = self.model.nx, self.model.ny
        for (j0, i0), (j1, i1) in zip(path[:-1], path[1:]):
            if i0 == i1 and abs(j1 - j0) == 1:
                j = min(j0, j1)
                i = i0
                if 0 < i < nx:
                    faces.append((name, i, j + 1, "X+"))
            elif j0 == j1 and abs(i1 - i0) == 1:
                j = j0
                i = min(i0, i1)
                if 0 < j < ny:
                    faces.append((name, i + 1, j, "Y+"))
            else:
                return []
        return faces

    def _optimized_raw_fault_faces(self, legacy_faces):
        base = self._base_pillar_array()
        result = []
        summary = {
            "enabled": True,
            "components": 0,
            "changed_components": 0,
            "legacy_faces": len(legacy_faces),
            "optimized_faces": 0,
            "legacy_cost": 0.0,
            "optimized_cost": 0.0,
            "fallback_components": 0,
            "preserved_components": 0,
        }
        eligible_names = set(self._conforming_fault_names())
        for name in self.model.fault_set.names:
            components = self._ordered_pillar_components_from_faces(
                legacy_faces, name
            )
            for chain in components:
                summary["components"] += 1
                if name not in eligible_names:
                    result.extend(self._faces_from_fault_pillar_path(name, chain))
                    summary["preserved_components"] += 1
                    continue
                trace = self._matched_top_trace_for_chain(name, chain, base)
                if trace is None:
                    result.extend(self._faces_from_fault_pillar_path(name, chain))
                    summary["fallback_components"] += 1
                    continue
                path, before, after = self._optimized_monotonic_fault_path(
                    base, trace, chain
                )
                faces = self._faces_from_fault_pillar_path(name, path)
                if not faces:
                    faces = self._faces_from_fault_pillar_path(name, chain)
                    path = list(chain)
                    after = before
                    summary["fallback_components"] += 1
                if path != list(chain):
                    summary["changed_components"] += 1
                summary["legacy_cost"] += float(before) if np.isfinite(before) else 0.0
                summary["optimized_cost"] += float(after) if np.isfinite(after) else 0.0
                result.extend(faces)

        result = sorted(set(result), key=lambda x: (x[0], x[2], x[1], x[3]))
        summary["optimized_faces"] = len(result)
        self._fault_path_optimization_summary = summary
        return result

    def _raw_fault_faces(self):
        """Discrete TOP fault path; V4 may optimize the v3.22 staircase.

        V3.22's centre-crossing path is always calculated first and remains the
        hard fallback.  V4 only changes a simple monotonic component when a
        connected edge path with the same endpoints has a strictly lower FLT
        proximity/alignment objective.
        """
        cached = getattr(self, "_raw_fault_faces_cache", None)
        if cached is not None:
            return cached
        if not self.model.split_faults or not self.model.fault_set:
            self._raw_fault_faces_cache = []
            self._fault_path_optimization_summary = {"enabled": False}
            return []

        legacy = self._legacy_raw_fault_faces()
        if bool(getattr(self.model, "fault_path_optimization", False)):
            chosen = self._optimized_raw_fault_faces(legacy)
            summary = getattr(self, "_fault_path_optimization_summary", {})
            print(
                "PyGRID V4: optimized fault-path search: "
                f"{int(summary.get('components', 0))} component(s), "
                f"{int(summary.get('changed_components', 0))} changed, "
                f"{int(summary.get('fallback_components', 0))} fallback, "
                f"{int(summary.get('preserved_components', 0))} preserved; "
                f"faces {len(legacy)} -> {len(chosen)}."
            )
            self._raw_fault_faces_cache = chosen
        else:
            self._fault_path_optimization_summary = {
                "enabled": False,
                "legacy_faces": len(legacy),
                "optimized_faces": len(legacy),
            }
            self._raw_fault_faces_cache = legacy
        return self._raw_fault_faces_cache

    @staticmethod
    def _fault_edge_endpoints(i1, j1, face):
        """Return zero-based pillar endpoints for one 1-based Eclipse face."""
        if face == "X+":
            return ((j1 - 1, i1), (j1, i1))
        if face == "Y+":
            return ((j1, i1 - 1), (j1, i1))
        return ()

    def _ordered_fault_pillar_components(self, name):
        """Return ordered pillar chains for one selected logical fault name."""
        return self._ordered_pillar_components_from_faces(
            self._raw_fault_faces(), name
        )

    def _matched_top_trace_for_chain(self, name, chain, base):
        """Return the TOP polyline belonging to one logical fault chain.

        V3.10 restricted conforming anchors to TOP traces that also had a
        matching FLB.  That made unmatched faults stay stair-stepped even when
        TinyECL ZigZag was OFF.

        V3.12 deliberately matches against *all* same-name TOP traces.  FLB is
        irrelevant for the TOP conforming construction; it is consulted later
        only when PyGRID decides whether the conformed fault should slant with
        depth.  Repeated same-name TOP pieces are still resolved geometrically
        by choosing the trace nearest the logical chain.
        """
        top_traces = self.model.fault_set.named_traces(name, bottom=False)
        if not top_traces or not chain:
            return None
        sample = np.mean(
            [np.asarray(base[j, i], dtype=float) for j, i in chain], axis=0
        )
        best = None
        for top_trace in top_traces:
            hit = _project_point_to_trace(sample, top_trace)
            if best is None or float(hit[1]) < best[0]:
                best = (float(hit[1]), top_trace)
        return None if best is None else best[1]

    def _fault_chain_anchors(self, base, names=None):
        """Map raw staircase vertices monotonically onto each continuous FLT.

        The mapping is deliberately by cumulative chain distance rather than by
        independently snapping every corner.  Thus a diagonal fault becomes a
        smooth sequence of edge chords and cannot fold back at each X/Y switch.
        """
        anchors_by_name = {}
        if names is None:
            names = self._conforming_fault_names()
        else:
            names = set(names)

        # COORD/ZigZag OFF conforms every TOP name.  MIXED/ZigZag ON conforms
        # only names that also have a matched exported BOTTOM trace.
        for name in self.model.fault_set.names:
            if name not in names:
                continue
            anchors = {}
            for chain in self._ordered_fault_pillar_components(name):
                if len(chain) < 2:
                    continue
                trace = self._matched_top_trace_for_chain(name, chain, base)
                if trace is None:
                    continue

                raw = np.asarray([base[j, i] for j, i in chain], dtype=float)
                seg = np.linalg.norm(raw[1:] - raw[:-1], axis=1)
                cumulative = np.concatenate(([0.0], np.cumsum(seg)))
                total = float(cumulative[-1])
                if total <= 1.0e-12:
                    continue

                s_first = float(_project_point_to_trace(raw[0], trace)[2])
                s_last = float(_project_point_to_trace(raw[-1], trace)[2])
                # Endpoints can occasionally project onto nearly the same bend.
                # Fall back to the full projected station spread in that case.
                projected = np.asarray(
                    [_project_point_to_trace(point, trace)[2] for point in raw],
                    dtype=float,
                )
                if abs(s_last - s_first) < 1.0e-4:
                    lo = float(np.min(projected))
                    hi = float(np.max(projected))
                    if hi - lo <= 1.0e-4:
                        continue
                    if projected[-1] >= projected[0]:
                        s_first, s_last = lo, hi
                    else:
                        s_first, s_last = hi, lo

                for key, distance in zip(chain, cumulative):
                    fraction = distance / total
                    station = s_first + fraction * (s_last - s_first)
                    target, tangent = _point_at_trace_station(trace, station)
                    anchors[key] = {
                        "target": np.asarray(target, dtype=float),
                        "station": float(station),
                        "trace": trace,
                        "tangent": np.asarray(tangent, dtype=float),
                    }
            if anchors:
                anchors_by_name[name] = anchors
        return anchors_by_name

    @staticmethod
    def _smooth_taper(value):
        value = min(1.0, max(0.0, float(value)))
        return 0.5 * (1.0 + np.cos(np.pi * value))

    def _propagate_fault_anchor_displacements(self, base, anchors_by_name, lines):
        """Build grid lines transverse to the interpreted fault.

        This is the second-generation fault-conforming construction.  The
        staircase remains only the *logical* Eclipse topology.  Geometrically,
        each fault anchor supplies the local FLT tangent and PyGRID rebuilds the
        logical row/column crossing that anchor in the direction normal to the
        continuous fault trace.

        A logical row is used when its I direction is closer to the fault
        normal; otherwise the J column is used.  The choice can therefore
        change while marching along a curved fault.  Displacement is tapered
        over ``lines`` pillars on both sides of the fault, so distant grid lines
        keep their original position.  Exact fault anchors are imposed again at
        the end and are never smoothed away.
        """
        base = np.asarray(base, dtype=float)
        ny, nx = self.model.ny, self.model.nx
        lines = max(1, int(lines))

        def unit(vector):
            vector = np.asarray(vector, dtype=float)
            length = float(np.linalg.norm(vector))
            if length <= 1.0e-12:
                return None
            return vector / length

        def local_axis(j, i, axis):
            if axis == "I":
                if i == 0:
                    vector = base[j, 1] - base[j, 0]
                elif i == nx:
                    vector = base[j, nx] - base[j, nx - 1]
                else:
                    vector = base[j, i + 1] - base[j, i - 1]
            else:
                if j == 0:
                    vector = base[1, i] - base[0, i]
                elif j == ny:
                    vector = base[ny, i] - base[ny - 1, i]
                else:
                    vector = base[j + 1, i] - base[j - 1, i]
            return unit(vector)

        # Exact targets.  At a true fault intersection several traces may ask
        # for one pillar; use their mean physical position.
        exact = {}
        for anchors in anchors_by_name.values():
            for key, data in anchors.items():
                exact.setdefault(key, []).append(
                    np.asarray(data["target"], dtype=float)
                )
        exact = {
            key: np.mean(np.asarray(values, dtype=float), axis=0)
            for key, values in exact.items()
        }

        numerator = np.zeros_like(base)
        weights = np.zeros((ny + 1, nx + 1), dtype=float)
        used_rows = set()
        used_columns = set()
        transverse_segments = []

        for anchors in anchors_by_name.values():
            # Process in along-fault order.  This makes the row/column switching
            # deterministic at staircase corners.
            ordered = sorted(
                anchors.items(),
                key=lambda item: float(item[1].get("station", 0.0)),
            )
            # V3 keeps the V2 construction but gently regularises only the
            # local transverse direction.  Fault-anchor positions remain exact.
            # Neighbouring tangents are sign-aligned and averaged 1:2:1 so the
            # row/column strips do not kink sharply from one station to the next.
            prepared = []
            for key, data in ordered:
                tangent = unit(data.get("tangent", (0.0, 0.0)))
                prepared.append((key, data, tangent))

            previous_axis = None
            for index, ((j, i), data, tangent) in enumerate(prepared):
                target = np.asarray(data["target"], dtype=float)
                if tangent is None:
                    continue

                tangent_sum = 2.0 * tangent
                tangent_weight = 2.0
                for neighbour_index in (index - 1, index + 1):
                    if not (0 <= neighbour_index < len(prepared)):
                        continue
                    neighbour = prepared[neighbour_index][2]
                    if neighbour is None:
                        continue
                    if float(np.dot(neighbour, tangent)) < 0.0:
                        neighbour = -neighbour
                    tangent_sum += neighbour
                    tangent_weight += 1.0
                tangent = unit(tangent_sum / tangent_weight)
                if tangent is None:
                    continue

                normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
                normal = unit(normal)
                if normal is None:
                    continue

                e_i = local_axis(j, i, "I")
                e_j = local_axis(j, i, "J")
                if e_i is None or e_j is None:
                    continue

                # The transverse logical line is whichever grid axis already
                # points most nearly normal to the continuous fault.  V3 adds
                # a small hysteresis around the I/J tie: keep the previous
                # transverse family unless the alternative is clearly better.
                score_i = abs(float(np.dot(normal, e_i)))
                score_j = abs(float(np.dot(normal, e_j)))
                preferred_axis = "I" if score_i >= score_j else "J"
                if previous_axis is not None and preferred_axis != previous_axis:
                    previous_score = score_i if previous_axis == "I" else score_j
                    preferred_score = score_i if preferred_axis == "I" else score_j
                    if preferred_score - previous_score < 0.08:
                        preferred_axis = previous_axis
                axis = preferred_axis
                previous_axis = axis
                transverse_segments.append({"key": (j, i), "axis": axis})
                if axis == "I":
                    forward = e_i
                    used_rows.add(j)
                else:
                    forward = e_j
                    used_columns.add(i)

                # Give the physical normal the same sign as increasing logical
                # index, so signed distances are stable on rotated grids.
                if float(np.dot(normal, forward)) < 0.0:
                    normal = -normal

                # Re-form the crossing line at 90 degrees to FLT.  Distances
                # are the original cumulative grid-edge lengths, preserving
                # local cell width rather than inventing a new spacing.
                for offset in range(-lines, lines + 1):
                    if axis == "I":
                        jj, ii = j, i + offset
                        if not (0 <= ii <= nx):
                            continue
                        if offset == 0:
                            signed_distance = 0.0
                        elif offset > 0:
                            signed_distance = sum(
                                float(np.linalg.norm(base[j, k + 1] - base[j, k]))
                                for k in range(i, ii)
                            )
                        else:
                            signed_distance = -sum(
                                float(np.linalg.norm(base[j, k] - base[j, k - 1]))
                                for k in range(i, ii, -1)
                            )
                    else:
                        jj, ii = j + offset, i
                        if not (0 <= jj <= ny):
                            continue
                        if offset == 0:
                            signed_distance = 0.0
                        elif offset > 0:
                            signed_distance = sum(
                                float(np.linalg.norm(base[k + 1, i] - base[k, i]))
                                for k in range(j, jj)
                            )
                        else:
                            signed_distance = -sum(
                                float(np.linalg.norm(base[k, i] - base[k - 1, i]))
                                for k in range(j, jj, -1)
                            )

                    ideal = target + normal * signed_distance
                    # Cosine taper: exact at the fault; zero at the edge of the
                    # conforming band.  Use lines+1 so the outermost requested
                    # line still receives a small, smooth correction.
                    weight = self._smooth_taper(abs(offset) / float(lines + 1))
                    if weight <= 1.0e-12:
                        continue

                    delta = ideal - base[jj, ii]
                    numerator[jj, ii] += weight * delta
                    weights[jj, ii] += weight

        result = base.copy()
        mask = weights > 1.0e-12
        if np.any(mask):
            # With a single transverse construction the cosine taper must stay
            # visible, hence divide by max(1,sum(w)) rather than normalising a
            # lone weight back to unity.  Where constructions overlap, this
            # automatically turns into a weighted average.
            denom = np.maximum(1.0, weights[mask])[:, None]
            result[mask] = base[mask] + numerator[mask] / denom

        # Keep the outer model footprint fixed unless the fault itself reaches
        # that boundary at an exact anchor.
        for j in range(ny + 1):
            for i in range(nx + 1):
                key = (j, i)
                if key not in exact and (j in (0, ny) or i in (0, nx)):
                    result[j, i] = base[j, i]

        # The fault edge is authoritative.
        for key, target in exact.items():
            result[key] = target

        self._conforming_transverse_rows = sorted(used_rows)
        self._conforming_transverse_columns = sorted(used_columns)
        self._conforming_transverse_segments = transverse_segments
        return result

    def _straighten_conforming_transverse_lines(
        self, base, result, anchors_by_name, lines, segments=None
    ):
        """Straighten the logical grid lines that cross a conforming FLT.

        ``_propagate_fault_anchor_displacements`` deliberately tapers each
        transverse reconstruction back into the untouched logical grid.  Where
        several reconstructions overlap, that taper can leave the visible
        row/column beside the fault slightly bowed even though the fault anchor
        itself is correct.

        This pass keeps every exact FLT anchor fixed and replaces only the
        affected row/column pieces by straight chords.  The chord controls are
        the exact anchors plus the first pillars just outside the conforming
        band.  If two transverse families cross, use the intersection of their
        chord lines so both remain straight.  The existing TOP topology guard
        runs afterwards and may relax non-anchor pillars if straightening would
        otherwise fold a cell.
        """
        base = np.asarray(base, dtype=float)
        result = np.asarray(result, dtype=float)
        lines = max(1, int(lines))
        if segments is None:
            segments = getattr(self, "_conforming_transverse_segments", ())
        if not segments:
            return result

        ny, nx = self.model.ny, self.model.nx
        exact = set()
        for anchors in anchors_by_name.values():
            exact.update(anchors.keys())

        # Group transverse anchors by their logical line.  I means a row
        # (vary I at fixed J); J means a column (vary J at fixed I).
        grouped = {}
        for item in segments:
            j, i = item["key"]
            axis = str(item["axis"]).upper()
            if axis == "I":
                grouped.setdefault(("I", j), set()).add(i)
            elif axis == "J":
                grouped.setdefault(("J", i), set()).add(j)

        proposals = {}

        def point_at(axis, fixed, pos):
            return result[fixed, pos] if axis == "I" else result[pos, fixed]

        def base_point_at(axis, fixed, pos):
            return base[fixed, pos] if axis == "I" else base[pos, fixed]

        def key_at(axis, fixed, pos):
            return (fixed, pos) if axis == "I" else (pos, fixed)

        def edge_length(axis, fixed, p0, p1):
            return float(np.linalg.norm(
                base_point_at(axis, fixed, p1) -
                base_point_at(axis, fixed, p0)
            ))

        # Build straight piecewise chords for each connected conforming band.
        # The first unaffected pillar is lines+1 indices from the anchor.
        reach = lines + 1
        for (axis, fixed), anchor_positions in grouped.items():
            limit = nx if axis == "I" else ny
            intervals = sorted(
                (max(0, p - reach), min(limit, p + reach))
                for p in anchor_positions
            )
            if not intervals:
                continue

            merged = []
            for lo, hi in intervals:
                if not merged or lo > merged[-1][1]:
                    merged.append([lo, hi])
                else:
                    merged[-1][1] = max(merged[-1][1], hi)

            for lo, hi in merged:
                # Every exact FLT anchor on this logical line is a hard control,
                # even if its own preferred transverse family is the other axis.
                controls = {lo, hi}
                for pos in range(lo, hi + 1):
                    if key_at(axis, fixed, pos) in exact:
                        controls.add(pos)
                controls = sorted(controls)

                for c0, c1 in zip(controls[:-1], controls[1:]):
                    if c1 <= c0 + 1:
                        continue
                    a = np.asarray(point_at(axis, fixed, c0), dtype=float)
                    b = np.asarray(point_at(axis, fixed, c1), dtype=float)
                    direction = b - a
                    if float(np.linalg.norm(direction)) <= 1.0e-12:
                        continue

                    cumulative = [0.0]
                    for pos in range(c0, c1):
                        cumulative.append(
                            cumulative[-1] + edge_length(axis, fixed, pos, pos + 1)
                        )
                    total = cumulative[-1]
                    if total <= 1.0e-12:
                        continue

                    for pos in range(c0 + 1, c1):
                        key = key_at(axis, fixed, pos)
                        if key in exact:
                            continue
                        fraction = cumulative[pos - c0] / total
                        target = a + fraction * direction
                        proposals.setdefault(key, []).append(
                            (target, a, b, axis)
                        )

        if not proposals:
            return result

        def line_intersection(a0, a1, b0, b1):
            p = np.asarray(a0, dtype=float)
            r = np.asarray(a1, dtype=float) - p
            q = np.asarray(b0, dtype=float)
            s = np.asarray(b1, dtype=float) - q
            denom = float(r[0] * s[1] - r[1] * s[0])
            scale = max(float(np.linalg.norm(r)), float(np.linalg.norm(s)), 1.0)
            if abs(denom) <= 1.0e-12 * scale * scale:
                return None
            qp = q - p
            t = float(qp[0] * s[1] - qp[1] * s[0]) / denom
            return p + t * r

        candidate = result.copy()
        for key, items in proposals.items():
            if key in exact:
                continue
            if len(items) == 1:
                candidate[key] = items[0][0]
                continue

            # Normally one row and one column proposal meet here.  Their line
            # intersection satisfies both straightness constraints exactly.
            first = items[0]
            second = next(
                (item for item in items[1:] if item[3] != first[3]), None
            )
            if second is not None:
                crossing = line_intersection(
                    first[1], first[2], second[1], second[2]
                )
                if crossing is not None and np.all(np.isfinite(crossing)):
                    candidate[key] = crossing
                    continue

            candidate[key] = np.mean(
                np.asarray([item[0] for item in items], dtype=float), axis=0
            )

        # Exact fault geometry is authoritative.
        for anchors in anchors_by_name.values():
            for key, data in anchors.items():
                candidate[key] = np.asarray(data["target"], dtype=float)

        return candidate

    @staticmethod
    def _signed_plan_cell_areas(pillars):
        """Signed XY area for every logical cell of one pillar grid."""
        p = np.asarray(pillars, dtype=float)
        a = p[:-1, :-1]
        b = p[:-1, 1:]
        c = p[1:, 1:]
        d = p[1:, :-1]
        return 0.5 * (
            (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])
            + (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])
            + (c[..., 0] * d[..., 1] - c[..., 1] * d[..., 0])
            + (d[..., 0] * a[..., 1] - d[..., 1] * a[..., 0])
        )

    def _guard_conformed_top_topology(self, base, result, anchors_by_name):
        """Untangle TOP plan cells without moving exact FLT anchors.

        Fine or strongly rotated grids can occasionally make the transverse
        deformation around an otherwise correct continuous FLT fold one plan
        cell.  The fault edge itself must remain authoritative, so this guard
        freezes every exact FLT anchor and reduces only the deformation of
        neighbouring non-anchor pillars toward the undeformed logical grid.

        If a cell is still invalid when all movable neighbours have returned
        essentially to their base positions, the problem lies in the anchor
        topology itself; in that case fail with a precise diagnostic rather
        than silently moving the fault off FLT.
        """
        if not bool(getattr(
            self.model, "slanted_fault_top_topology_guard", True
        )):
            return result

        base = np.asarray(base, dtype=float)
        result = np.asarray(result, dtype=float)
        displacement = result - base

        base_areas = self._signed_plan_cell_areas(base)
        finite = base_areas[np.isfinite(base_areas)]
        if len(finite) == 0:
            return result
        orientation = 1.0 if float(np.median(finite)) >= 0.0 else -1.0
        base_signed = orientation * base_areas
        positive = base_signed[np.isfinite(base_signed) & (base_signed > 0.0)]
        if len(positive) == 0:
            return result
        if np.any(~np.isfinite(base_signed) | (base_signed <= 0.0)):
            raise ValueError(
                "Undeformed logical TOP grid itself contains invalid plan cells."
            )

        representative = float(np.median(positive))
        minimum_ratio = max(0.0, float(getattr(
            self.model, "slanted_fault_top_min_area_ratio", 0.0
        )))
        minimum_area = np.maximum(
            minimum_ratio * base_signed,
            max(1.0e-12 * representative, 1.0e-30),
        )

        frozen = np.zeros(base.shape[:2], dtype=bool)
        for anchors in anchors_by_name.values():
            for j, i in anchors:
                if 0 <= j < frozen.shape[0] and 0 <= i < frozen.shape[1]:
                    frozen[j, i] = True

        factor = float(getattr(
            self.model, "slanted_fault_top_repair_factor", 0.85
        ))
        if not (0.0 < factor < 1.0):
            factor = 0.85
        passes = max(1, int(getattr(
            self.model, "slanted_fault_top_repair_passes", 64
        )))

        candidate = result.copy()
        areas = orientation * self._signed_plan_cell_areas(candidate)
        initial_bad = int(np.count_nonzero(
            ~np.isfinite(areas) | (areas < minimum_area)
        ))
        if initial_bad == 0:
            return result

        ny, nx = self.model.ny, self.model.nx
        scale = np.ones((ny + 1, nx + 1), dtype=float)

        for _pass in range(passes):
            areas = orientation * self._signed_plan_cell_areas(candidate)
            bad = np.argwhere(~np.isfinite(areas) | (areas < minimum_area))
            if len(bad) == 0:
                break

            touched = set()
            for j, i in bad:
                touched.update(((j, i), (j, i + 1),
                                (j + 1, i), (j + 1, i + 1)))

            changed = False
            for j, i in touched:
                if frozen[j, i]:
                    continue
                if float(np.linalg.norm(displacement[j, i])) <= 1.0e-9:
                    continue
                scale[j, i] *= factor
                changed = True
            if not changed:
                break

            candidate = base + displacement * scale[..., None]
            # Exact FLT anchors are immutable.  Re-impose them explicitly so a
            # future refactor cannot accidentally let the local scale touch them.
            candidate[frozen] = result[frozen]

        final_areas = orientation * self._signed_plan_cell_areas(candidate)
        remaining_mask = ~np.isfinite(final_areas) | (final_areas < minimum_area)
        remaining = int(np.count_nonzero(remaining_mask))
        changed_mask = (~frozen) & (scale < (1.0 - 1.0e-12))
        changed_count = int(np.count_nonzero(changed_mask))
        retained = float(np.min(scale[changed_mask])) if changed_count else 1.0
        final_ratio = float(np.nanmin(
            final_areas / np.maximum(base_signed, 1.0e-30)
        ))

        if remaining:
            worst = np.argwhere(remaining_mask)
            wj, wi = map(int, worst[0])
            corners = ((wj, wi), (wj, wi + 1),
                       (wj + 1, wi), (wj + 1, wi + 1))
            fixed_count = sum(bool(frozen[j, i]) for j, i in corners)
            movable_count = sum(
                (not bool(frozen[j, i]))
                and float(np.linalg.norm(displacement[j, i])) > 1.0e-9
                for j, i in corners
            )
            raise ValueError(
                "TOP fault-conforming grid cannot preserve positive plan "
                f"topology while keeping FLT anchors exact ({remaining} cell(s) "
                f"remain; first I={wi + 1},J={wj + 1}; "
                f"fault-anchor corners={fixed_count}, movable deformed "
                f"corners={movable_count})."
            )

        print(
            "PyGRID: TOP topology guard: repaired "
            f"{initial_bad} weak/inverted plan cell(s) by reducing transverse "
            f"deformation at {changed_count} non-anchor pillars; exact FLT "
            f"anchors preserved; minimum retained transverse deformation "
            f"fraction = {retained:.3f}; minimum TOP/base plan-area ratio = "
            f"{final_ratio:.6f}."
        )
        return candidate

    def _guard_conformed_bottom_topology(self, base, result):
        """Locally reduce lower-grid deformation before it can fold cells.

        FLB is a requested lower fault trace, but a sufficiently fine logical
        grid can make the full FLT->FLB displacement geometrically impossible
        in a few cells.  V3.14 keeps the V3.13 TOP geometry untouched and
        reduces only the BOTTOM displacement of pillars touching cells whose
        signed plan area would fall below a small fraction of a normal cell.

        The repair is deliberately local: unaffected FLB anchors and lower-grid
        pillars retain 100 percent of their requested displacement.
        """
        if not bool(getattr(
            self.model, "slanted_fault_bottom_topology_guard", True
        )):
            return result

        base = np.asarray(base, dtype=float)
        result = np.asarray(result, dtype=float)
        displacement = result - base
        if not np.any(np.linalg.norm(displacement, axis=2) > 1.0e-9):
            return result

        base_areas = self._signed_plan_cell_areas(base)
        finite = base_areas[np.isfinite(base_areas)]
        if len(finite) == 0:
            return result
        orientation = 1.0 if float(np.median(finite)) >= 0.0 else -1.0
        base_signed = orientation * base_areas
        positive = base_signed[np.isfinite(base_signed) & (base_signed > 0.0)]
        if len(positive) == 0:
            return result

        representative = float(np.median(positive))
        minimum_ratio = max(
            0.0,
            float(getattr(
                self.model, "slanted_fault_bottom_min_area_ratio", 0.02
            )),
        )

        # V3.16: the topology floor must be local to the already-valid TOP
        # conforming cell, not a fixed fraction of the median grid-cell area.
        # On a fine/rotated grid a legitimate TOP cell can be much smaller than
        # the median.  A global floor can then become impossible to satisfy even
        # when the BOTTOM displacement has been reduced all the way back to the
        # TOP geometry.  Require each lower cell to retain a small fraction of
        # *its own* TOP plan area instead.
        invalid_base = (~np.isfinite(base_signed)) | (base_signed <= 0.0)
        if np.any(invalid_base):
            count = int(np.count_nonzero(invalid_base))
            raise ValueError(
                "TOP fault-conforming grid is not positive before BOTTOM "
                f"deformation ({count} invalid plan cells)."
            )
        absolute_epsilon = max(1.0e-12 * representative, 1.0e-30)
        minimum_area = np.maximum(
            minimum_ratio * base_signed,
            absolute_epsilon,
        )

        factor = float(getattr(
            self.model, "slanted_fault_bottom_repair_factor", 0.90
        ))
        if not (0.0 < factor < 1.0):
            factor = 0.90
        passes = max(1, int(getattr(
            self.model, "slanted_fault_bottom_repair_passes", 24
        )))

        ny, nx = self.model.ny, self.model.nx
        scale = np.ones((ny + 1, nx + 1), dtype=float)
        candidate = result.copy()
        initial_areas = orientation * self._signed_plan_cell_areas(candidate)
        initial_bad = int(np.count_nonzero(
            ~np.isfinite(initial_areas) | (initial_areas < minimum_area)
        ))
        if initial_bad == 0:
            return result

        for _pass in range(passes):
            areas = orientation * self._signed_plan_cell_areas(candidate)
            bad = np.argwhere(
                ~np.isfinite(areas) | (areas < minimum_area)
            )
            if len(bad) == 0:
                break

            touched = set()
            for j, i in bad:
                touched.update(((j, i), (j, i + 1),
                                (j + 1, i), (j + 1, i + 1)))
            changed = False
            for j, i in touched:
                if float(np.linalg.norm(displacement[j, i])) <= 1.0e-9:
                    continue
                scale[j, i] *= factor
                changed = True
            if not changed:
                break
            candidate = base + displacement * scale[..., None]

        final_areas = orientation * self._signed_plan_cell_areas(candidate)
        remaining = int(np.count_nonzero(
            ~np.isfinite(final_areas) | (final_areas < minimum_area)
        ))
        changed_mask = scale < (1.0 - 1.0e-12)
        changed_count = int(np.count_nonzero(changed_mask))
        retained = (
            float(np.min(scale[changed_mask])) if changed_count else 1.0
        )
        local_ratios = final_areas / np.maximum(base_signed, 1.0e-30)
        final_min_ratio = float(np.nanmin(local_ratios))

        if remaining:
            worst = np.argwhere(
                ~np.isfinite(final_areas) | (final_areas < minimum_area)
            )
            if len(worst):
                wj, wi = map(int, worst[0])
                detail = (
                    f" first remaining cell I={wi + 1},J={wj + 1}; "
                    f"BOTTOM/TOP plan-area ratio="
                    f"{local_ratios[wj, wi]:.6f}"
                )
            else:
                detail = ""
            raise ValueError(
                "BOTTOM fault-conforming grid cannot preserve positive cell "
                f"topology after {passes} local repair passes "
                f"({remaining} cells remain below their local area floor)." +
                detail
            )

        print(
            "PyGRID: BOTTOM topology guard: repaired "
            f"{initial_bad} weak/inverted plan cells by reducing lower-grid "
            f"deformation at {changed_count} local pillars; minimum retained "
            f"FLT->FLB displacement fraction = {retained:.3f}; "
            f"minimum BOTTOM/TOP plan-area ratio = {final_min_ratio:.4f}."
        )
        return candidate

    @staticmethod
    def _grid_spacing_from_array(pillars):
        p = np.asarray(pillars, dtype=float)
        values = []
        if p.shape[1] > 1:
            values.extend(np.linalg.norm(p[:, 1:] - p[:, :-1], axis=2).ravel())
        if p.shape[0] > 1:
            values.extend(np.linalg.norm(p[1:] - p[:-1], axis=2).ravel())
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values) & (values > 1.0e-9)]
        return float(np.median(values)) if len(values) else 1.0

    def _conformed_top_pillar_xy(self, names=None):
        if names is None:
            names = self._conforming_fault_names()
        names = tuple(sorted(set(names)))
        cached = getattr(self, "_conformed_top_pillar_xy_cache", None)
        cached_names = getattr(self, "_conformed_top_pillar_xy_cache_names", None)
        if cached is not None and cached_names == names:
            return cached
        base = self._base_pillar_array()
        anchors = self._fault_chain_anchors(base, names=names)
        lines = max(1, int(getattr(self.model, "slanted_fault_conform_lines", 4)))
        result = self._propagate_fault_anchor_displacements(base, anchors, lines)
        result = self._straighten_conforming_transverse_lines(
            base, result, anchors, lines
        )
        result = self._guard_conformed_top_topology(base, result, anchors)
        self._conformed_fault_top_anchors = anchors
        self._conformed_top_pillar_xy_cache = result
        self._conformed_top_pillar_xy_cache_names = names
        moved = int(np.count_nonzero(np.linalg.norm(result - base, axis=2) > 1.0e-9))
        anchor_count = sum(len(item) for item in anchors.values())
        label = ",".join(names) if names else "none"
        print(
            "PyGRID: fault-conforming TOP grid: "
            f"{anchor_count} fault-edge pillars snapped to FLT; "
            f"{moved} pillars reshaped through the grid interior "
            f"(faults: {label})."
        )
        return result

    def _fault_set_for_level(self, level):
        """Return the fault barriers appropriate to one structural endpoint.

        TOP is controlled by TinyECL ``*.flt``.  BOTTOM is controlled by the
        matched ``*.flb`` interpretation where that lower trace has geometric
        support.  If FLB is shorter than FLT, unsupported FLT tails remain at
        the top-trace XY and therefore stay vertical instead of disappearing in
        the last K layer.  This is the key TOP/BOTTOM-first construction rule:
        the two endpoint horizons are built independently before intermediate
        layers are filled.
        """
        level = str(level).lower()
        if level not in {"top", "bottom"}:
            raise ValueError("level must be 'top' or 'bottom'")

        if not self.model.split_faults or not self.model.fault_set:
            return FaultSet()

        if level == "top":
            return self.model.fault_set

        if level in self._surface_fault_sets_cache:
            return self._surface_fault_sets_cache[level]

        # A same-name FLB may describe only part of a longer FLT trace.
        # Replacing the complete top trace by the shorter bottom polyline makes
        # the fault topology jump sideways (or disappear) in the last K layer.
        # Build the BOTTOM trace with the same support-aware top-down migration
        # used by intermediate LAYERED faults.  Unsupported FLT tails therefore
        # remain vertical, while the supported part migrates toward FLB.
        result = self._migrated_fault_set(1.0)
        self._surface_fault_sets_cache[level] = result
        return result

    def _slanted_fault_mode(self):
        """Return the effective TinyECL/PyGRID fault-geometry mode."""
        if hasattr(self.model, "effective_slanted_fault_mode"):
            return self.model.effective_slanted_fault_mode()

        # Backward compatibility for older stand-alone GridModel objects.
        mode = str(
            getattr(self.model, "slanted_fault_mode", "LAYERED") or "LAYERED"
        ).upper()
        if mode not in {"LAYERED", "MIXED", "COORD"}:
            raise ValueError(
                "model.slanted_fault_mode must be 'LAYERED', 'MIXED' or 'COORD'"
            )
        return mode

    def _migrated_fault_set(self, fraction):
        """Return support-aware FLT->FLB fault traces at one depth fraction.

        The top trace remains the topology anchor.  Where a matched FLB really
        supports that top piece, points migrate by the stable slant displacement.
        Where FLB is shorter or absent, the FLT position is retained.  This
        prevents a shorter FLB from making the last K layer jump across several
        grid blocks.
        """
        fraction = min(1.0, max(0.0, float(fraction)))

        if not self.model.split_faults or not self.model.fault_set:
            return FaultSet()

        spacing = self._grid_spacing()
        traces = []

        for top_trace in self.model.fault_set.traces:
            points = np.asarray(top_trace.points, dtype=float)
            if len(points) == 0:
                continue

            migrated = []
            for point in points:
                displacement = np.asarray(
                    self.model.fault_set.slant_displacement(
                        top_trace.name,
                        point,
                        max_top_distance=1.5 * spacing,
                        fade_distance=2.0 * spacing,
                    ),
                    dtype=float,
                )
                migrated.append(point + fraction * displacement)

            traces.append(
                FaultTrace(
                    name=top_trace.name,
                    source_id=top_trace.source_id,
                    points=np.asarray(migrated, dtype=float),
                    surface_barrier=top_trace.surface_barrier,
                    surface_crossings=top_trace.surface_crossings,
                )
            )

        return FaultSet(traces=traces)

    def _fault_set_for_fraction(self, fraction):
        """Return a fault trace set for one vertical layer fraction.

        In LAYERED mode the grid pillars remain vertical.  A slanted fault is
        represented instead by a stair-step trace that migrates horizontally
        from the TinyECL FLT interpretation at fraction 0 to the matched FLB
        interpretation at fraction 1.  Unmatched/vertical fault pieces remain
        fixed at their FLT position.

        The trace interpolation deliberately uses the already-stable
        ``slant_displacement`` correspondence, so repeated same-name pieces and
        shorter FLB segments keep the safeguards from the slanted-fault work.
        """
        fraction = float(fraction)
        fraction = min(1.0, max(0.0, fraction))

        if (
            self._slanted_fault_mode() != "LAYERED"
            or not self.model.split_faults
            or not self.model.fault_set
        ):
            return self._fault_set_for_level("top")

        if fraction <= 1.0e-12:
            return self._fault_set_for_level("top")
        if fraction >= 1.0 - 1.0e-12:
            return self._fault_set_for_level("bottom")

        cache = getattr(self, "_fraction_fault_sets_cache", None)
        if cache is None:
            cache = {}
            self._fraction_fault_sets_cache = cache

        key = round(fraction, 12)
        if key in cache:
            return cache[key]

        result = self._migrated_fault_set(fraction)
        cache[key] = result
        return result

    @staticmethod
    def _fault_set_cache_key(fault_set):
        if not fault_set:
            return ()
        return tuple(
            (id(trace), bool(getattr(trace, "surface_barrier", True)))
            for trace in fault_set.traces
        )

    def _fault_pillar_names(self):
        """Map grid-pillar indices to the source fault names touching them."""
        result = {}
        slanted = set(self.model.fault_set.slanted_names)
        if not slanted:
            return result

        for name, i1, j1, face in self.fault_faces():
            if name not in slanted:
                continue

            # fault_faces uses 1-based Eclipse cell indices.
            if face == "X+":
                # East edge of cell (i1,j1): pillar column i1.
                endpoints = ((j1 - 1, i1), (j1, i1))
            elif face == "Y+":
                # South edge of cell (i1,j1): pillar row j1.
                endpoints = ((j1, i1 - 1), (j1, i1))
            else:
                continue

            for key in endpoints:
                result.setdefault(key, set()).add(name)

        return result

    def _grid_spacing(self):
        """Return a robust representative horizontal grid-pillar spacing."""
        if hasattr(self, "_grid_spacing_cache"):
            return self._grid_spacing_cache

        p = self._pillar_array()
        values = []
        if p.shape[1] > 1:
            values.extend(np.linalg.norm(p[:, 1:] - p[:, :-1], axis=2).ravel())
        if p.shape[0] > 1:
            values.extend(np.linalg.norm(p[1:] - p[:-1], axis=2).ravel())

        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values) & (values > 1.0e-9)]
        if len(values) == 0:
            result = 1.0
        else:
            result = float(np.median(values))

        self._grid_spacing_cache = result
        return result

    def _fault_reference_dz(self, name):
        """Robust top-to-bottom structural depth separation for ``name``."""
        if not hasattr(self, "_fault_reference_dz_cache"):
            self._fault_reference_dz_cache = {}
        if name in self._fault_reference_dz_cache:
            return self._fault_reference_dz_cache[name]

        top_map = getattr(self.model, "top_map", None)
        bottom_map = getattr(self.model, "bottom_map", None)
        if top_map is None or bottom_map is None:
            self._fault_reference_dz_cache[name] = None
            return None

        spacing = self._grid_spacing()
        values = []

        # Only matched top/bottom groups define slanted geometry.  This avoids
        # letting an unrelated repeated same-name top polyline borrow another
        # group's bottom trace.
        for top_trace, _bottom_trace in self.model.fault_set.matched_trace_pairs(name):
            for point in top_trace.points:
                point = np.asarray(point, dtype=float)
                displacement = self.model.fault_set.slant_displacement(
                    name,
                    point,
                    max_top_distance=1.5 * spacing,
                    fade_distance=2.0 * spacing,
                )
                if float(np.linalg.norm(displacement)) <= 1.0e-9:
                    continue

                bottom_xy = point + displacement
                z_top = float(
                    np.asarray(
                        top_map.interpolate(point), dtype=float
                    ).reshape(-1)[0]
                )
                z_bottom = float(
                    np.asarray(
                        bottom_map.interpolate(bottom_xy), dtype=float
                    ).reshape(-1)[0]
                )
                dz = z_bottom - z_top
                if np.isfinite(dz) and abs(dz) > 1.0e-6:
                    values.append(dz)

        if not values:
            result = None
        else:
            result = float(np.median(np.asarray(values, dtype=float)))

        self._fault_reference_dz_cache[name] = result
        return result

    def _fault_reference_depths(self, name, top_xy, bottom_xy):
        """Return stable structural top/bottom depths for a slanted pillar.

        Scattered-map interpolation can occasionally make the local TOP and
        BOTTOM surfaces nearly coincide or reverse at a fault.  Such a tiny
        denominator would create an unrealistically long COORD stick.  Use the
        median separation measured along the named source traces as a robust
        fallback for these local outliers.
        """
        top_map = getattr(self.model, "top_map", None)
        bottom_map = getattr(self.model, "bottom_map", None)
        thickness_map = getattr(self.model, "thickness_map", None)

        if top_map is None:
            return None

        z_top = float(
            np.asarray(
                top_map.interpolate(np.asarray(top_xy, dtype=float)), dtype=float
            ).reshape(-1)[0]
        )

        if bottom_map is not None:
            z_bottom = float(
                np.asarray(
                    bottom_map.interpolate(np.asarray(bottom_xy, dtype=float)),
                    dtype=float,
                ).reshape(-1)[0]
            )
            local_dz = z_bottom - z_top
            reference_dz = self._fault_reference_dz(name)

            if reference_dz is not None and abs(reference_dz) > 1.0e-6:
                ratio = abs(local_dz) / abs(reference_dz)
                if (
                    local_dz * reference_dz <= 0.0
                    or ratio < 0.25
                    or ratio > 3.0
                ):
                    z_bottom = z_top + reference_dz

        elif thickness_map is not None:
            thickness = float(
                thickness_map.interpolate(np.asarray(top_xy, dtype=float))
            )
            if abs(thickness) <= 1.0e-6:
                return None
            z_bottom = z_top + thickness

        else:
            return None

        if abs(z_bottom - z_top) <= 1.0e-6:
            return None
        return z_top, z_bottom

    def _slanted_trace_projection(self, name, point):
        """Return nearest matched top trace and projection for ``point``.

        A fault name may occur in more than one disconnected top polyline.
        Only a top polyline that has a geometrically matched bottom polyline is
        eligible for slanted geometry.  This mirrors ``slant_displacement`` and
        gives the staircase-spike filter a stable along-fault station.
        """
        point = np.asarray(point, dtype=float)
        best = None

        for top_trace, _bottom_trace in self.model.fault_set.matched_trace_pairs(name):
            hit = _project_point_to_trace(point, top_trace)
            distance = float(hit[1])
            if best is None or distance < best[0]:
                best = (distance, top_trace, hit)

        if best is None:
            return None
        return best[1], best[2]

    def _equalize_slanted_fault_spikes(self, records_by_name):
        """Remove only isolated stair-corner spikes from slanted fault sticks.

        A stair-stepped Eclipse fault changes between X and Y faces.  At such a
        turn two different grid pillars can project to almost the same station
        on the interpreted fault trace.  Normally their structural depths are
        similar.  Occasionally one pillar samples the wrong local side of a
        strongly faulted surface and becomes a narrow triangular spike.

        The correction is intentionally local and conservative:

        * only slanted faults are considered;
        * only two-or-more pillars within a quarter grid spacing *along the
          same matched source trace* are candidate staircase twins;
        * a candidate needs valid neighbours before and after it;
        * one twin must be a clear outlier relative to the local along-fault
          depth trend;
        * only that bad stick is replaced by the good twin's local
          displacement/top/bottom depths.

        Thus ordinary fault throw, fault dip and the rest of the fault plane
        remain unchanged.
        """
        if not bool(getattr(self.model, "equalize_slanted_fault_spikes", True)):
            return

        grid_spacing = self._grid_spacing()
        corner_fraction = float(
            getattr(self.model, "slanted_spike_corner_fraction", 0.25)
        )
        error_ratio = float(
            getattr(self.model, "slanted_spike_error_ratio", 1.75)
        )
        relative_jump = float(
            getattr(self.model, "slanted_spike_relative_jump", 0.20)
        )

        if corner_fraction <= 0.0:
            return

        for name, records in records_by_name.items():
            # Treat disconnected same-name source traces independently.
            by_trace = {}
            for record in records:
                if record.get("depths") is None:
                    continue
                trace = record.get("top_trace")
                if trace is None:
                    continue
                by_trace.setdefault(id(trace), []).append(record)

            for trace_records in by_trace.values():
                if len(trace_records) < 4:
                    continue

                top_trace = trace_records[0]["top_trace"]
                trace_length = max(float(top_trace.length), 1.0e-12)
                station_tolerance = corner_fraction * grid_spacing / trace_length

                ordered = sorted(trace_records, key=lambda item: item["station"])

                # Build clusters of pillars that occupy effectively the same
                # along-fault station.  Typical normal station increments are
                # about one grid spacing, so the tolerance is deliberately much
                # smaller than that.
                clusters = []
                cluster = [0]
                for idx in range(1, len(ordered)):
                    if (
                        ordered[idx]["station"] - ordered[idx - 1]["station"]
                        <= station_tolerance
                    ):
                        cluster.append(idx)
                    else:
                        if len(cluster) >= 2:
                            clusters.append(cluster)
                        cluster = [idx]
                if len(cluster) >= 2:
                    clusters.append(cluster)

                for indices in clusters:
                    first = indices[0]
                    last = indices[-1]

                    # We need the local trend on both sides.  Endpoint clusters
                    # are therefore left untouched; this also preserves the
                    # existing smooth taper where a bottom trace terminates.
                    if first == 0 or last >= len(ordered) - 1:
                        continue

                    left = ordered[first - 1]
                    right = ordered[last + 1]
                    if left.get("depths") is None or right.get("depths") is None:
                        continue

                    station = float(
                        np.mean([ordered[idx]["station"] for idx in indices])
                    )
                    span = right["station"] - left["station"]
                    if span <= 1.0e-12:
                        continue
                    t = (station - left["station"]) / span

                    expected = np.asarray(
                        [
                            left["z_fault_top"]
                            + t * (right["z_fault_top"] - left["z_fault_top"]),
                            left["z_fault_bottom"]
                            + t * (
                                right["z_fault_bottom"] - left["z_fault_bottom"]
                            ),
                        ],
                        dtype=float,
                    )

                    candidate_records = [ordered[idx] for idx in indices]
                    errors = []
                    for record in candidate_records:
                        actual = np.asarray(
                            [record["z_fault_top"], record["z_fault_bottom"]],
                            dtype=float,
                        )
                        errors.append(float(np.linalg.norm(actual - expected)))

                    if len(errors) < 2:
                        continue

                    good_pos = int(np.argmin(errors))
                    bad_pos = int(np.argmax(errors))
                    if good_pos == bad_pos:
                        continue

                    good_error = errors[good_pos]
                    bad_error = errors[bad_pos]
                    local_zone = abs(float(expected[1] - expected[0]))

                    # Scale-free conservative trigger.  The bad twin must be
                    # materially worse than the good twin and the excess error
                    # must be significant compared with the local structural
                    # top-to-bottom separation.
                    if bad_error < error_ratio * max(good_error, 1.0e-9):
                        continue
                    if (
                        bad_error - good_error
                        < relative_jump * max(local_zone, 1.0e-9)
                    ):
                        continue

                    good = candidate_records[good_pos]
                    bad = candidate_records[bad_pos]

                    bad["displacement"] = np.asarray(
                        good["displacement"], dtype=float
                    ).copy()
                    bad["z_fault_top"] = float(good["z_fault_top"])
                    bad["z_fault_bottom"] = float(good["z_fault_bottom"])
                    bad["depths"] = (
                        bad["z_fault_top"],
                        bad["z_fault_bottom"],
                    )
                    bad["equalized_from"] = good["key"]

                    station_distance = station * trace_length
                    print(
                        "PyGRID: equalized slanted-fault corner spike: "
                        f"{name} near station {station_distance:.1f} "
                        f"({bad['key']} -> {good['key']})."
                    )

    def _relax_slanted_fault_records(self, records_by_key, base):
        """Spread lower-fault movement over neighbouring grid lines.

        ZigZag mode keeps the structural TOP on the untouched regular grid.
        Only the lower end of a slanted COORD stick is displaced.  Moving only
        the pillars directly on a fault squeezes or stretches one row of cells;
        therefore the full fault displacement is tapered linearly across a
        configurable band of neighbouring grid lines on both sides.

        A large top-to-bottom shift automatically gets a wider band so the
        change between adjacent lower-grid lines remains a reasonable fraction
        of one normal block width.  If a top-fault pillar has no valid matching
        bottom-fault point it never becomes an anchor and remains straight.
        """
        if not bool(getattr(self.model, "relax_slanted_faults", True)):
            return records_by_key

        min_lines = max(0, int(getattr(self.model, "slanted_fault_relax_lines", 4)))
        max_step_fraction = float(
            getattr(self.model, "slanted_fault_max_step_fraction", 0.35)
        )
        if min_lines <= 0:
            return records_by_key
        if max_step_fraction <= 0.0:
            max_step_fraction = 0.35

        ny, nx = self.model.ny, self.model.nx
        spacing = self._grid_spacing()

        # One physical lower-grid displacement per staircase pillar.  At a
        # true intersection of two slanted faults use the same mean convention
        # as the final COORD writer.
        anchors = []
        for key, records in records_by_key.items():
            valid = [
                record for record in records
                if float(np.linalg.norm(record["displacement"])) > 1.0e-9
            ]
            if not valid:
                continue

            displacement = np.mean(
                [np.asarray(record["displacement"], dtype=float) for record in valid],
                axis=0,
            )
            dominant = max(
                valid,
                key=lambda record: float(np.linalg.norm(record["displacement"])),
            )
            magnitude = float(np.linalg.norm(displacement))

            # If the fault shift is large, widen the relaxation band rather
            # than allowing one or two very narrow lower cells.
            auto_lines = int(
                np.ceil(magnitude / max(max_step_fraction * spacing, 1.0e-12))
            )
            lines = max(min_lines, auto_lines)

            anchors.append(
                {
                    "key": key,
                    "displacement": displacement,
                    "name": dominant["name"],
                    "lines": lines,
                }
            )

        if not anchors:
            return records_by_key

        contributions = {}
        for anchor in anchors:
            aj, ai = anchor["key"]
            lines = anchor["lines"]
            displacement = anchor["displacement"]

            # radius is lines+1 because line 0 (the fault itself) has weight 1
            # and the first untouched grid line outside the band has weight 0.
            radius = float(lines + 1)
            for dj in range(-lines, lines + 1):
                j = aj + dj
                if j < 0 or j > ny:
                    continue
                for di in range(-lines, lines + 1):
                    i = ai + di
                    if i < 0 or i > nx:
                        continue

                    distance = float(np.hypot(di, dj))
                    if distance > lines:
                        continue
                    weight = max(0.0, 1.0 - distance / radius)
                    if weight <= 0.0:
                        continue

                    contributions.setdefault((j, i), []).append(
                        (weight, displacement, anchor["name"], distance)
                    )

        result = {key: list(records) for key, records in records_by_key.items()}
        relaxed_count = 0

        for key, values in contributions.items():
            # Direct fault pillars keep their full already-corrected slant.
            if key in records_by_key:
                continue

            weights = np.asarray([item[0] for item in values], dtype=float)
            vectors = np.asarray([item[1] for item in values], dtype=float)
            if not np.any(weights > 0.0):
                continue

            # Average along the fault where neighbouring anchor bands overlap,
            # then apply the strongest cross-fault taper at this grid pillar.
            mean_vector = np.sum(vectors * weights[:, None], axis=0) / np.sum(weights)
            taper = float(np.max(weights))
            displacement = mean_vector * taper
            if float(np.linalg.norm(displacement)) <= 1.0e-9:
                continue

            # Use the nearest anchor's fault name only for the robust depth
            # separation estimate.  Geometry itself comes from the blended
            # displacement field above.
            nearest = min(values, key=lambda item: item[3])
            name = nearest[2]
            j, i = key
            base_xy = np.asarray(base[j, i], dtype=float)
            desired_bottom_xy = base_xy + displacement
            depths = self._fault_reference_depths(name, base_xy, desired_bottom_xy)

            record = {
                "key": key,
                "name": name,
                "base_xy": base_xy.copy(),
                "displacement": np.asarray(displacement, dtype=float),
                "top_trace": None,
                "station": None,
                "depths": depths,
                "z_fault_top": None,
                "z_fault_bottom": None,
                "zigzag_relaxed": True,
            }
            if depths is not None:
                record["z_fault_top"] = float(depths[0])
                record["z_fault_bottom"] = float(depths[1])

            result[key] = [record]
            relaxed_count += 1

        if relaxed_count:
            widest = max(anchor["lines"] for anchor in anchors)
            print(
                "PyGRID: ZigZag slanted faults: relaxed lower-grid cell widths "
                f"over up to {widest} grid lines ({relaxed_count} neighbouring pillars)."
            )

        return result

    @staticmethod
    def _record_coord_endpoints(record):
        """Convert one prepared slanted-fault record to two COORD points."""
        base_xy = np.asarray(record["base_xy"], dtype=float)
        displacement = np.asarray(record["displacement"], dtype=float)
        depths = record.get("depths")

        if depths is None:
            return (
                np.asarray([base_xy[0], base_xy[1], 0.0], dtype=float),
                np.asarray(
                    [
                        base_xy[0] + displacement[0],
                        base_xy[1] + displacement[1],
                        100000.0,
                    ],
                    dtype=float,
                ),
            )

        z_fault_top = float(record["z_fault_top"])
        z_fault_bottom = float(record["z_fault_bottom"])
        dz = z_fault_bottom - z_fault_top
        if abs(dz) <= 1.0e-12:
            return None

        slope = displacement / dz
        margin = max(25.0, 0.20 * abs(dz))
        z1 = min(z_fault_top, z_fault_bottom) - margin
        z2 = max(z_fault_top, z_fault_bottom) + margin

        xy1 = base_xy + slope * (z1 - z_fault_top)
        xy2 = base_xy + slope * (z2 - z_fault_top)

        return (
            np.asarray([xy1[0], xy1[1], z1], dtype=float),
            np.asarray([xy2[0], xy2[1], z2], dtype=float),
        )

    def _coord_pillars(self):
        """Return COORD pillars for the selected slanted-fault representation.

        LAYERED (default)
        -----------------
        Every pillar is vertical and keeps the original regular-grid XY from
        TOP to BOTTOM.  This deliberately restores the geometry of PyGRID's
        first non-slanted implementation: at any Z, the intersection with a
        pillar has the same XY.  Therefore TOP cell footprints stay rectangular
        even where the two sides of a fault have different ZCORN depths.

        The pillar Z endpoints simply bracket all adjacent ZCORN interfaces.
        They do not control the surface shape; ZCORN remains authoritative.

        COORD (legacy experimental mode)
        --------------------------------
        Preserve the previous top/bottom-endpoint slant for comparison.
        """
        base = self._pillar_array()
        cached = getattr(self, "_coord_pillars_cache", None)
        if cached is not None:
            return cached[0].copy(), cached[1].copy()

        ny, nx = self.model.ny, self.model.nx

        if self._slanted_fault_mode() == "LAYERED":
            coord_top = np.empty((ny + 1, nx + 1, 3), dtype=float)
            coord_bottom = np.empty((ny + 1, nx + 1, 3), dtype=float)

            interfaces = None
            try:
                interfaces = self._cell_corner_interfaces()
            except ValueError:
                interfaces = None

            for j in range(ny + 1):
                for i in range(nx + 1):
                    xy = np.asarray(base[j, i], dtype=float)

                    if interfaces is None:
                        z1 = self.PILLAR_Z_TOP
                        z2 = self.PILLAR_Z_BOTTOM
                    else:
                        values = []
                        for cj, ci, corner in self._pillar_adjacent_corner_indices(
                            j, i, ny, nx
                        ):
                            values.extend(
                                np.asarray(
                                    interfaces[:, cj, ci, corner], dtype=float
                                ).tolist()
                            )

                        values = np.asarray(values, dtype=float)
                        values = values[np.isfinite(values)]
                        if len(values) == 0:
                            z1 = self.PILLAR_Z_TOP
                            z2 = self.PILLAR_Z_BOTTOM
                        else:
                            z1 = float(np.min(values))
                            z2 = float(np.max(values))
                            if abs(z2 - z1) <= 1.0e-9:
                                z2 = z1 + 1.0

                    coord_top[j, i] = (xy[0], xy[1], z1)
                    coord_bottom[j, i] = (xy[0], xy[1], z2)

            return coord_top, coord_bottom

        # True COORD slant: orient inclined pillars from a ruled fault plane.
        # The XY FLT->FLB displacement defines the plane generator, while the
        # generator slope uses a robust structural TOP->BOTTOM depth separation
        # from the interpreted surfaces.  Do not derive the slope from the
        # median adjacent ZCORN depths: at a fault those medians can sit on
        # opposite throw blocks and make the pillar artificially several times
        # too steep.
        bottom_xy = self._bottom_pillar_xy()
        coord_top = np.empty((ny + 1, nx + 1, 3), dtype=float)
        coord_bottom = np.empty((ny + 1, nx + 1, 3), dtype=float)

        # V3.15 keeps enough information to steepen only a local COORD pillar
        # if the complete 3-D corner grid would otherwise invert.  Scaling is
        # performed about the TOP fault-plane pivot so FLT conformity remains
        # authoritative while only the local dip is reduced.
        coord_pivot_xy = np.full((ny + 1, nx + 1, 2), np.nan, dtype=float)
        coord_pivot_z = np.full((ny + 1, nx + 1), np.nan, dtype=float)
        coord_slope_xy = np.zeros((ny + 1, nx + 1, 2), dtype=float)
        coord_slant_mask = np.zeros((ny + 1, nx + 1), dtype=bool)

        interfaces = None
        try:
            interfaces = self._cell_corner_interfaces()
        except ValueError:
            interfaces = None

        slanted_names = list(getattr(self.model.fault_set, "slanted_names", []))
        capped_count = 0
        max_requested_slope = 0.0

        for j in range(ny + 1):
            for i in range(nx + 1):
                top_xy = np.asarray(base[j, i], dtype=float)
                lower_xy = np.asarray(bottom_xy[j, i], dtype=float)
                displacement = lower_xy - top_xy

                values = []
                if interfaces is not None:
                    for cj, ci, corner in self._pillar_adjacent_corner_indices(
                        j, i, ny, nx
                    ):
                        values.extend(
                            np.asarray(
                                interfaces[:, cj, ci, corner], dtype=float
                            ).tolist()
                        )
                values = np.asarray(values, dtype=float)
                values = values[np.isfinite(values)]

                if len(values) == 0:
                    z_min = self.PILLAR_Z_TOP
                    z_max = self.PILLAR_Z_BOTTOM
                else:
                    z_min = float(np.min(values))
                    z_max = float(np.max(values))
                    if abs(z_max - z_min) <= 1.0e-9:
                        z_max = z_min + 1.0

                if float(np.linalg.norm(displacement)) <= 1.0e-9:
                    coord_top[j, i] = (top_xy[0], top_xy[1], z_min)
                    coord_bottom[j, i] = (top_xy[0], top_xy[1], z_max)
                    continue

                # Associate a relaxed/slanted pillar with the nearest matched
                # fault piece only to obtain its robust plane depth interval.
                nearest_name = None
                nearest_distance = None
                for name in slanted_names:
                    projection = self._slanted_trace_projection(name, top_xy)
                    if projection is None:
                        continue
                    distance = float(projection[1][1])
                    if nearest_distance is None or distance < nearest_distance:
                        nearest_distance = distance
                        nearest_name = name

                plane_depths = None
                if nearest_name is not None:
                    plane_depths = self._fault_reference_depths(
                        nearest_name, top_xy, lower_xy
                    )

                if plane_depths is None:
                    # Conservative fallback: preserve the former endpoint rule.
                    coord_top[j, i] = (top_xy[0], top_xy[1], z_min)
                    coord_bottom[j, i] = (lower_xy[0], lower_xy[1], z_max)
                    fallback_dz = float(z_max - z_min)
                    if abs(fallback_dz) > 1.0e-9:
                        coord_pivot_xy[j, i] = top_xy
                        coord_pivot_z[j, i] = z_min
                        coord_slope_xy[j, i] = displacement / fallback_dz
                        coord_slant_mask[j, i] = True
                    continue

                z_fault_top = float(plane_depths[0])
                z_fault_bottom = float(plane_depths[1])
                dz = z_fault_bottom - z_fault_top
                if abs(dz) <= 1.0e-9:
                    coord_top[j, i] = (top_xy[0], top_xy[1], z_min)
                    coord_bottom[j, i] = (lower_xy[0], lower_xy[1], z_max)
                    fallback_dz = float(z_max - z_min)
                    if abs(fallback_dz) > 1.0e-9:
                        coord_pivot_xy[j, i] = top_xy
                        coord_pivot_z[j, i] = z_min
                        coord_slope_xy[j, i] = displacement / fallback_dz
                        coord_slant_mask[j, i] = True
                    continue

                slope = displacement / dz

                # Work in one physical unit system before applying the local
                # pillar-slope safety limit.  Internal PyGRID XY may be METRES
                # while Z is FEET, so the raw slope above can be mixed-unit.
                xy_to_grid = self._unit_factor(self._map_units(), self._grid_units())
                physical_slope = slope * xy_to_grid
                requested_slope = float(np.linalg.norm(physical_slope))
                max_slope = float(
                    getattr(self.model, "slanted_fault_max_pillar_slope", 1.5)
                )
                conforming = bool(
                    getattr(self.model, "conform_slanted_faults", False)
                )
                adaptive_coord_guard = bool(
                    getattr(self.model, "slanted_fault_coord_topology_guard", True)
                )

                # A conforming fault already has an adaptive 3-D topology
                # guard below.  Do not globally steepen every COORD stick
                # before that guard runs: on coarse grids the old H/V cap
                # reduced different fault stations by very different amounts
                # and produced the visible accordion/faceted fault ribbon.
                # Keep the legacy global cap only for non-conforming COORD
                # geometry, or when the adaptive guard is explicitly disabled.
                if conforming and adaptive_coord_guard:
                    max_slope = 0.0
                elif conforming:
                    conform_limit = float(
                        getattr(
                            self.model,
                            "slanted_fault_conform_max_pillar_slope",
                            1.0,
                        )
                    )
                    if conform_limit > 0.0:
                        if max_slope <= 0.0:
                            max_slope = conform_limit
                        else:
                            max_slope = min(max_slope, conform_limit)

                if max_slope > 0.0 and requested_slope > max_slope:
                    physical_slope *= max_slope / requested_slope
                    slope = physical_slope / xy_to_grid
                    capped_count += 1
                    max_requested_slope = max(max_requested_slope, requested_slope)

                coord_pivot_xy[j, i] = top_xy
                coord_pivot_z[j, i] = z_fault_top
                coord_slope_xy[j, i] = slope
                coord_slant_mask[j, i] = (
                    float(np.linalg.norm(slope)) > 1.0e-12
                )

                # Bracket every adjacent ZCORN point so the reader normally
                # interpolates on, rather than extrapolates far beyond, COORD.
                margin = max(25.0, 0.05 * max(abs(dz), abs(z_max - z_min)))
                z1 = min(z_min, z_fault_top, z_fault_bottom) - margin
                z2 = max(z_max, z_fault_top, z_fault_bottom) + margin
                xy1 = top_xy + slope * (z1 - z_fault_top)
                xy2 = top_xy + slope * (z2 - z_fault_top)

                coord_top[j, i] = (xy1[0], xy1[1], z1)
                coord_bottom[j, i] = (xy2[0], xy2[1], z2)

        if capped_count:
            max_slope = float(
                getattr(self.model, "slanted_fault_max_pillar_slope", 1.5)
            )
            if bool(getattr(self.model, "conform_slanted_faults", False)):
                conform_limit = float(
                    getattr(
                        self.model,
                        "slanted_fault_conform_max_pillar_slope",
                        1.0,
                    )
                )
                if conform_limit > 0.0:
                    max_slope = (
                        conform_limit if max_slope <= 0.0
                        else min(max_slope, conform_limit)
                    )
            print(
                "PyGRID: slanted fault-plane orientation: limited "
                f"{capped_count} local pillar slopes to H/V <= {max_slope:g}; "
                f"largest requested H/V slope was {max_requested_slope:.3f}."
            )

        coord_top, coord_bottom = self._guard_slanted_coord_topology(
            coord_top,
            coord_bottom,
            interfaces,
            coord_pivot_xy,
            coord_pivot_z,
            coord_slope_xy,
            coord_slant_mask,
        )

        coord_top, coord_bottom = self._align_fault_coord_outliers(
            coord_top, coord_bottom
        )
        v41_seed_top = coord_top.copy()
        v41_seed_bottom = coord_bottom.copy()
        coord_top, coord_bottom = self._conform_fault_trace_band(
            coord_top, coord_bottom
        )
        coord_top, coord_bottom = self._straighten_fault_plane_generators(
            coord_top, coord_bottom
        )
        coord_top, coord_bottom = self._redistribute_fault_trace_band_harmonic(
            v41_seed_top, v41_seed_bottom, coord_top, coord_bottom
        )
        self._coord_pillars_cache = (coord_top.copy(), coord_bottom.copy())
        return coord_top, coord_bottom


    @staticmethod
    def _coord_points_at_depths(coord_top, coord_bottom, z):
        """Vectorized equivalent of _point_on_coord_pillar for many depths."""
        top = np.asarray(coord_top, dtype=float)
        bottom = np.asarray(coord_bottom, dtype=float)
        z = np.asarray(z, dtype=float)
        dz = bottom[..., 2] - top[..., 2]
        safe_dz = np.where(np.abs(dz) > 1.0e-12, dz, 1.0)
        fraction = (z - top[..., 2][None, ...]) / safe_dz[None, ...]
        dx = (bottom[..., 0] - top[..., 0])[None, ...]
        dy = (bottom[..., 1] - top[..., 1])[None, ...]
        x = top[..., 0][None, ...] + fraction * dx
        y = top[..., 1][None, ...] + fraction * dy
        vertical = np.abs(dz) <= 1.0e-12
        if np.any(vertical):
            x[:, vertical] = top[..., 0][vertical]
            y[:, vertical] = top[..., 1][vertical]
        return np.stack((x, y, z), axis=-1)

    def _cell_center_jacobians_from_coords(self, coord_top, coord_bottom, interfaces=None):
        """Centre Jacobians for explicitly supplied COORD arrays.

        V3.15 evaluates the same trilinear centre-Jacobian as the original
        triple Python loop, but vectorizes it over all I/J/K cells.  This makes
        local adaptive COORD repair practical on fine grids without changing
        the QC definition.
        """
        if interfaces is None:
            interfaces = self._cell_corner_interfaces()

        corner_slices = (
            (slice(None, -1), slice(None, -1)),
            (slice(None, -1), slice(1, None)),
            (slice(1, None), slice(None, -1)),
            (slice(1, None), slice(1, None)),
        )
        top_vertices = []
        bottom_vertices = []
        for corner, grid_slice in enumerate(corner_slices):
            pillar_top = np.asarray(coord_top[grid_slice], dtype=float)
            pillar_bottom = np.asarray(coord_bottom[grid_slice], dtype=float)
            top_vertices.append(
                self._coord_points_at_depths(
                    pillar_top,
                    pillar_bottom,
                    interfaces[:-1, ..., corner],
                )
            )
            bottom_vertices.append(
                self._coord_points_at_depths(
                    pillar_top,
                    pillar_bottom,
                    interfaces[1:, ..., corner],
                )
            )

        v000, v100, v010, v110 = top_vertices
        v001, v101, v011, v111 = bottom_vertices
        di = (
            (v100 + v110 + v101 + v111)
            - (v000 + v010 + v001 + v011)
        ) / 4.0
        dj = (
            (v010 + v110 + v011 + v111)
            - (v000 + v100 + v001 + v101)
        ) / 4.0
        dk = (
            (v001 + v101 + v011 + v111)
            - (v000 + v100 + v010 + v110)
        ) / 4.0
        return np.einsum("...i,...i->...", di, np.cross(dj, dk))

    def _pinched_cell_mask(self, interfaces=None):
        """Return K/J/I mask for intentionally collapsed ZCORN cells.

        A cell is considered pinched only when all four ZCORN corner
        thicknesses are effectively zero.  This distinguishes a legitimate
        structural pinchout from a negative/non-finite Jacobian caused by
        invalid COORD geometry.
        """
        if interfaces is None:
            interfaces = self._cell_corner_interfaces()
        values = np.asarray(interfaces, dtype=float)
        if values.ndim != 4 or values.shape[0] < 2 or values.shape[-1] != 4:
            return np.zeros(values.shape[1:4] if values.ndim >= 4 else (0,), dtype=bool)

        dz = values[1:] - values[:-1]
        finite = np.all(np.isfinite(dz), axis=3)
        positive = dz[np.isfinite(dz) & (dz > 0.0)]
        representative = float(np.median(positive)) if positive.size else 1.0
        tolerance = max(1.0e-12, 1.0e-9 * max(representative, 1.0))
        return finite & np.all(np.abs(dz) <= tolerance, axis=3)


    def _guard_slanted_coord_topology(
        self,
        coord_top,
        coord_bottom,
        interfaces,
        pivot_xy,
        pivot_z,
        slope_xy,
        slant_mask,
    ):
        """Steepen only local COORD pillars needed to remove 3-D inversions.

        V3.14 guarantees a positive TOP/BOTTOM plan-view topology.  A fine grid
        can nevertheless invert in 3-D because a straight inclined COORD stick
        cuts through the strongly varying ZCORN geometry.  This guard leaves
        the already-conformed TOP fault position fixed and scales only the local
        XY/Z slope around its TOP fault-plane pivot.
        """
        if not bool(getattr(
            self.model, "slanted_fault_coord_topology_guard", True
        )):
            return coord_top, coord_bottom
        if interfaces is None:
            return coord_top, coord_bottom

        original_top = np.asarray(coord_top, dtype=float)
        original_bottom = np.asarray(coord_bottom, dtype=float)
        candidate_top = original_top.copy()
        candidate_bottom = original_bottom.copy()
        pivot_xy = np.asarray(pivot_xy, dtype=float)
        pivot_z = np.asarray(pivot_z, dtype=float)
        slope_xy = np.asarray(slope_xy, dtype=float)
        slant_mask = np.asarray(slant_mask, dtype=bool)
        valid_slant = (
            slant_mask
            & np.isfinite(pivot_z)
            & np.all(np.isfinite(pivot_xy), axis=2)
            & np.all(np.isfinite(slope_xy), axis=2)
        )
        if not np.any(valid_slant):
            return coord_top, coord_bottom

        dets = self._cell_center_jacobians_from_coords(
            candidate_top, candidate_bottom, interfaces
        )
        pinched = self._pinched_cell_mask(interfaces)
        active_finite = np.isfinite(dets) & ~pinched
        finite = dets[active_finite]
        if len(finite) == 0:
            return coord_top, coord_bottom
        median = float(np.median(finite))
        orientation = 1.0 if median >= 0.0 else -1.0
        median_abs = max(abs(median), 1.0e-30)
        signed = orientation * dets
        initial_bad_mask = (
            ~np.isfinite(signed)
            | ((signed <= 0.0) & ~pinched)
        )
        initial_inverted = int(np.count_nonzero(initial_bad_mask))
        if initial_inverted == 0:
            return coord_top, coord_bottom

        minimum_ratio = max(0.0, float(getattr(
            self.model, "slanted_fault_coord_min_jacobian_ratio", 0.005
        )))
        minimum_det = minimum_ratio * median_abs
        factor = float(getattr(
            self.model, "slanted_fault_coord_repair_factor", 0.90
        ))
        if not (0.0 < factor < 1.0):
            factor = 0.90
        passes = max(1, int(getattr(
            self.model, "slanted_fault_coord_repair_passes", 32
        )))

        ny, nx = self.model.ny, self.model.nx
        scale = np.ones((ny + 1, nx + 1), dtype=float)

        for _pass in range(passes):
            signed = orientation * self._cell_center_jacobians_from_coords(
                candidate_top, candidate_bottom, interfaces
            )
            bad_cells = np.any(
                ~np.isfinite(signed)
                | ((signed < minimum_det) & ~pinched),
                axis=0,
            )
            if not np.any(bad_cells):
                break

            touched = np.zeros((ny + 1, nx + 1), dtype=bool)
            touched[:-1, :-1] |= bad_cells
            touched[:-1, 1:] |= bad_cells
            touched[1:, :-1] |= bad_cells
            touched[1:, 1:] |= bad_cells
            active = touched & valid_slant
            if not np.any(active):
                break

            scale[active] *= factor

            # Rebuild each changed COORD line around the fixed TOP fault-plane
            # pivot.  Both endpoints move, but the line still passes through
            # exactly the same TOP conforming point at pivot_z.
            for endpoints in (candidate_top, candidate_bottom):
                dz = endpoints[..., 2] - pivot_z
                new_x = pivot_xy[..., 0] + (
                    slope_xy[..., 0] * scale * dz
                )
                new_y = pivot_xy[..., 1] + (
                    slope_xy[..., 1] * scale * dz
                )
                endpoints[..., 0][valid_slant] = new_x[valid_slant]
                endpoints[..., 1][valid_slant] = new_y[valid_slant]

        final_dets = self._cell_center_jacobians_from_coords(
            candidate_top, candidate_bottom, interfaces
        )
        final_signed = orientation * final_dets
        remaining_inverted = int(np.count_nonzero(
            ~np.isfinite(final_signed)
            | ((final_signed <= 0.0) & ~pinched)
        ))
        changed = valid_slant & (scale < (1.0 - 1.0e-12))
        changed_count = int(np.count_nonzero(changed))
        retained = float(np.min(scale[changed])) if changed_count else 1.0
        active_final = final_signed[~pinched & np.isfinite(final_signed)]
        final_ratio = (
            float(np.min(active_final)) / median_abs
            if active_final.size else 0.0
        )

        if remaining_inverted:
            print(
                "PyGRID: 3-D COORD topology guard: reduced local fault dip at "
                f"{changed_count} pillars, but {remaining_inverted} inverted "
                "cells remain for final QC."
            )
            return candidate_top, candidate_bottom

        print(
            "PyGRID: 3-D COORD topology guard: repaired "
            f"{initial_inverted} inverted cells by steepening "
            f"{changed_count} local fault pillars; minimum retained fault-plane "
            f"slant fraction = {retained:.3f}; minimum centre-Jacobian ratio = "
            f"{final_ratio:.4f}."
        )
        return candidate_top, candidate_bottom


    def _align_fault_coord_outliers(self, coord_top, coord_bottom):
        """Align only genuine local COORD outliers on the fault ribbon.

        V3 remains the geometric baseline: logical fault faces, the conforming
        TOP/BOTTOM grid and the surrounding deformation are not changed here.

        V3.4 proved that the remaining visible kinks can be detected as local
        departures from the smooth along-fault COORD ribbon.  V3.6 keeps that
        first correction unchanged, then performs a moderately stronger
        polishing stage.  V3.9 additionally scans past blocked upper-edge
        candidates instead of letting one QC-limited pillar stop the pass.  After the primary correction the ribbon is measured
        again; only the strongest residual mismatch is considered per pass and
        it is moved only part-way toward the local target.  Every trial must
        improve the ribbon *and* pass the complete corner-point Jacobian QC.
        """
        records = []
        if not bool(getattr(self.model, "align_fault_coord_outliers", True)):
            self._coord_alignment_records = records
            self._coord_alignment_summary = {}
            return coord_top, coord_bottom
        if not (
            bool(getattr(self.model, "conform_slanted_faults", False))
            and self._slanted_fault_mode() in {"COORD", "MIXED"}
            and self.model.split_faults
            and self.model.fault_set
            and self.model.fault_set.slanted_names
        ):
            self._coord_alignment_records = records
            self._coord_alignment_summary = {}
            return coord_top, coord_bottom

        spacing = self._grid_spacing()
        tolerance = max(
            1.0e-9,
            float(getattr(
                self.model, "fault_coord_alignment_tolerance_fraction", 0.12
            )) * spacing,
        )
        base = self._base_pillar_array()
        anchors_by_name = self._fault_chain_anchors(base)
        original_top = np.asarray(coord_top, dtype=float).copy()
        original_bottom = np.asarray(coord_bottom, dtype=float).copy()

        interfaces = self._cell_corner_interfaces()
        pinched = self._pinched_cell_mask(interfaces)
        baseline_dets = self._cell_center_jacobians_from_coords(
            original_top, original_bottom, interfaces
        )
        active_finite = np.isfinite(baseline_dets) & ~pinched
        finite = baseline_dets[active_finite]
        median = float(np.median(finite)) if len(finite) else 1.0
        orientation = 1.0 if median >= 0.0 else -1.0
        median_abs = max(abs(median), 1.0e-30)
        min_required = float(getattr(
            self.model, "fault_coord_alignment_min_jacobian_ratio", 0.02
        ))

        def measure(top_array, bottom_array):
            """Return every measurable interior fault-stick ribbon mismatch."""
            measured = []
            for name, anchors in anchors_by_name.items():
                ordered = sorted(
                    anchors.items(), key=lambda item: float(item[1]["station"])
                )
                values = []
                for key, data in ordered:
                    top_target = np.asarray(data["target"], dtype=float)
                    displacement = np.asarray(
                        self.model.fault_set.slant_displacement(
                            name,
                            top_target,
                            max_top_distance=1.5 * spacing,
                            fade_distance=2.0 * spacing,
                        ),
                        dtype=float,
                    )
                    if float(np.linalg.norm(displacement)) <= 1.0e-9:
                        values.append(None)
                        continue
                    bottom_target = top_target + displacement
                    depths = self._fault_reference_depths(
                        name, top_target, bottom_target
                    )
                    if depths is None:
                        values.append(None)
                        continue
                    z_top, z_bottom = map(float, depths)
                    if abs(z_bottom - z_top) <= 1.0e-9:
                        values.append(None)
                        continue
                    current_xy = self._point_on_coord_pillar(
                        top_array[key], bottom_array[key], z_bottom
                    )[:2]
                    values.append({
                        "key": key,
                        "name": name,
                        "station": float(data["station"]),
                        "top_target": top_target,
                        "z_top": z_top,
                        "z_bottom": z_bottom,
                        "current_xy": current_xy,
                        "offset": current_xy - top_target,
                    })

                for index in range(1, len(values) - 1):
                    cur = values[index]
                    prev = values[index - 1]
                    nxt = values[index + 1]
                    if cur is None or prev is None or nxt is None:
                        continue
                    denom = float(nxt["station"] - prev["station"])
                    if abs(denom) <= 1.0e-12:
                        continue
                    w = float((cur["station"] - prev["station"]) / denom)
                    z = cur["z_bottom"]
                    prev_xy = self._point_on_coord_pillar(
                        top_array[prev["key"]], bottom_array[prev["key"]], z
                    )[:2]
                    next_xy = self._point_on_coord_pillar(
                        top_array[nxt["key"]], bottom_array[nxt["key"]], z
                    )[:2]
                    prev_offset = prev_xy - prev["top_target"]
                    next_offset = next_xy - nxt["top_target"]
                    expected_offset = prev_offset + w * (
                        next_offset - prev_offset
                    )
                    delta = expected_offset - cur["offset"]
                    deviation = float(np.linalg.norm(delta))
                    measured.append({
                        **cur,
                        "expected_offset": expected_offset,
                        "delta": delta,
                        "deviation": deviation,
                    })
            return measured

        def quality(top_array, bottom_array):
            dets = self._cell_center_jacobians_from_coords(
                top_array, bottom_array, interfaces
            )
            signed = orientation * dets
            bad_mask = (
                ~np.isfinite(signed)
                | ((signed <= 0.0) & ~pinched)
            )
            bad = int(np.count_nonzero(bad_mask))
            active = signed[np.isfinite(signed) & ~pinched]
            min_ratio = (
                float(np.min(active)) / median_abs
                if len(active) else -np.inf
            )
            return bad, min_ratio

        def trial_move(top_array, bottom_array, item, fraction=1.0, max_shift=None):
            """Return trial arrays after a bounded move toward ``item`` target."""
            fraction = min(1.0, max(0.0, float(fraction)))
            delta = np.asarray(item["delta"], dtype=float) * fraction
            delta_length = float(np.linalg.norm(delta))
            if max_shift is not None and max_shift > 0.0 and delta_length > max_shift:
                delta *= float(max_shift) / delta_length

            key = item["key"]
            top_target = item["top_target"]
            z_top = item["z_top"]
            z_bottom = item["z_bottom"]
            target_xy = item["current_xy"] + delta
            slope = (target_xy - top_target) / (z_bottom - z_top)

            trial_top = np.asarray(top_array, dtype=float).copy()
            trial_bottom = np.asarray(bottom_array, dtype=float).copy()
            z1 = float(trial_top[key][2])
            z2 = float(trial_bottom[key][2])
            trial_top[key][:2] = top_target + slope * (z1 - z_top)
            trial_bottom[key][:2] = top_target + slope * (z2 - z_top)
            return trial_top, trial_bottom, target_xy

        # ------------------------------------------------------------------
        # Primary V3.4 correction: preserve the proven behaviour exactly.
        # Candidate detection is against untouched V3 geometry so one accepted
        # move cannot manufacture a second primary candidate.
        # ------------------------------------------------------------------
        primary_measured = measure(original_top, original_bottom)
        primary_candidates = [
            item for item in primary_measured if item["deviation"] > tolerance
        ]
        primary_candidates.sort(key=lambda item: item["deviation"], reverse=True)

        result_top = original_top.copy()
        result_bottom = original_bottom.copy()

        for item in primary_candidates:
            trial_top, trial_bottom, target_xy = trial_move(
                result_top, result_bottom, item, fraction=1.0
            )
            bad, min_ratio = quality(trial_top, trial_bottom)
            accepted = bad == 0 and min_ratio >= min_required

            if accepted:
                result_top = trial_top
                result_bottom = trial_bottom
                after_xy = self._point_on_coord_pillar(
                    result_top[item["key"]],
                    result_bottom[item["key"]],
                    item["z_bottom"],
                )[:2]
                deviation_after = float(np.linalg.norm(after_xy - target_xy))
                applied = float(np.linalg.norm(after_xy - item["current_xy"]))
            else:
                deviation_after = item["deviation"]
                applied = 0.0

            records.append({
                "phase": "PRIMARY",
                "pass": 0,
                "name": item["name"],
                "key": item["key"],
                "station": item["station"],
                "deviation_before": item["deviation"],
                "deviation_after": deviation_after,
                "applied_shift": applied,
                "accepted": accepted,
                "trial_min_jacobian_ratio": min_ratio,
            })

        # ------------------------------------------------------------------
        # V3.6 polish: re-measure after the primary corrections.  Work on one
        # strongest residual outlier at a time and move it only part-way.  The
        # trial must reduce both the maximum ribbon mismatch and the total
        # squared mismatch, in addition to passing the global Jacobian QC.
        # ------------------------------------------------------------------
        polish_passes = max(0, int(getattr(
            self.model, "fault_coord_polish_passes", 4
        )))
        polish_fraction = float(getattr(
            self.model, "fault_coord_polish_fraction", 0.80
        ))
        polish_fraction = min(1.0, max(0.0, polish_fraction))
        polish_max_shift = max(
            0.0,
            float(getattr(
                self.model, "fault_coord_polish_max_shift_fraction", 0.12
            )) * spacing,
        )

        for pass_number in range(1, polish_passes + 1):
            before_all = measure(result_top, result_bottom)
            if not before_all:
                break
            before_candidates = [
                item for item in before_all if item["deviation"] > tolerance
            ]
            if not before_candidates:
                break
            before_candidates.sort(
                key=lambda item: item["deviation"], reverse=True
            )
            item = before_candidates[0]
            before_max = max(x["deviation"] for x in before_all)
            before_sumsq = sum(x["deviation"] ** 2 for x in before_all)

            trial_top, trial_bottom, _target_xy = trial_move(
                result_top,
                result_bottom,
                item,
                fraction=polish_fraction,
                max_shift=polish_max_shift,
            )
            bad, min_ratio = quality(trial_top, trial_bottom)
            after_all = measure(trial_top, trial_bottom)
            after_lookup = {x["key"]: x for x in after_all}
            after_item = after_lookup.get(item["key"])
            deviation_after = (
                float(after_item["deviation"])
                if after_item is not None else float(item["deviation"])
            )
            after_max = (
                max(x["deviation"] for x in after_all)
                if after_all else 0.0
            )
            after_sumsq = sum(x["deviation"] ** 2 for x in after_all)

            improved = (
                deviation_after < item["deviation"] - 1.0e-9
                and after_max < before_max - 1.0e-9
                and after_sumsq < before_sumsq - 1.0e-9
            )
            accepted = (
                bad == 0
                and min_ratio >= min_required
                and improved
            )

            if accepted:
                before_xy = self._point_on_coord_pillar(
                    result_top[item["key"]],
                    result_bottom[item["key"]],
                    item["z_bottom"],
                )[:2]
                after_xy = self._point_on_coord_pillar(
                    trial_top[item["key"]],
                    trial_bottom[item["key"]],
                    item["z_bottom"],
                )[:2]
                applied = float(np.linalg.norm(after_xy - before_xy))
                result_top = trial_top
                result_bottom = trial_bottom
            else:
                applied = 0.0
                deviation_after = item["deviation"]

            records.append({
                "phase": "POLISH",
                "pass": pass_number,
                "name": item["name"],
                "key": item["key"],
                "station": item["station"],
                "deviation_before": item["deviation"],
                "deviation_after": deviation_after,
                "applied_shift": applied,
                "accepted": accepted,
                "trial_min_jacobian_ratio": min_ratio,
                "max_deviation_before": before_max,
                "max_deviation_after": after_max if accepted else before_max,
                "sumsq_before": before_sumsq,
                "sumsq_after": after_sumsq if accepted else before_sumsq,
            })
            if not accepted:
                # The strongest residual could not be improved safely.  Do not
                # work around it by moving weaker neighbours and risk a cascade.
                break

        # ------------------------------------------------------------------
        # V3.10 upper-edge alignment.  The lower fault edge is already good.
        # Measure the actual top fault-edge intersections of each existing
        # COORD stick directly against the matched FLT polyline.  Rotate only
        # the misaligned stick about its *current lower fault-edge point*.
        # This leaves topology and the good lower edge fixed while moving the
        # upper edge by the shortest XY distance onto the interpreted FLT.
        # ------------------------------------------------------------------
        face_depths = {}
        nz = int(self.model.layers)
        for face_name, i1, j1, face in self._raw_fault_faces():
            i0 = i1 - 1
            j0 = j1 - 1
            if face == "X+":
                endpoints = (((j0, i0 + 1), 1), ((j0 + 1, i0 + 1), 3))
            elif face == "Y+":
                endpoints = (((j0 + 1, i0), 2), ((j0 + 1, i0 + 1), 3))
            else:
                continue
            for key, corner in endpoints:
                bucket = face_depths.setdefault((face_name, key), {
                    "top": [], "bottom": []
                })
                bucket["top"].append(float(interfaces[0, j0, i0, corner]))
                bucket["bottom"].append(float(interfaces[nz, j0, i0, corner]))

        def measure_top_edge(top_array, bottom_array):
            """Measure the actual TOP fault-edge distance to FLT.

            V3.10 deliberately stops extrapolating a ruled FLT->FLB plane to
            the local top ZCORN depth.  The user's interpreted FLT polyline is
            the authoritative top-fault line.  For every existing logical
            fault pillar we evaluate the COORD stick at the real top fault-edge
            Z values, project those XY points to the matched FLT polyline, and
            use only that perpendicular XY residual as the requested move.

            The good lower fault edge remains the pivot.  Thus topology, cell
            identity and the lower edge are unchanged; only the inclination of
            a misaligned existing COORD stick is corrected toward FLT.
            """
            measured = []
            for name, anchors in anchors_by_name.items():
                for key, data in anchors.items():
                    depth_data = face_depths.get((name, key))
                    if not depth_data or not depth_data["top"] or not depth_data["bottom"]:
                        continue
                    trace = data.get("trace")
                    if trace is None:
                        continue
                    top_zs = np.unique(np.asarray(depth_data["top"], dtype=float))
                    bottom_zs = np.unique(np.asarray(depth_data["bottom"], dtype=float))
                    pivot_z = float(np.median(bottom_zs))
                    residuals = []
                    dzs = []
                    errors = []
                    projected_stations = []
                    for z in top_zs:
                        current_xy = self._point_on_coord_pillar(
                            top_array[key], bottom_array[key], float(z)
                        )[:2]
                        hit = _project_point_to_trace(current_xy, trace)
                        desired_xy = np.asarray(hit[0], dtype=float)
                        residual = desired_xy - current_xy
                        residuals.append(residual)
                        dzs.append(float(z) - pivot_z)
                        errors.append(float(hit[1]))
                        projected_stations.append(float(hit[2]))
                    denom = float(np.sum(np.asarray(dzs, dtype=float) ** 2))
                    if denom <= 1.0e-12:
                        continue
                    correction_slope = np.sum(
                        np.asarray(dzs, dtype=float)[:, None] *
                        np.asarray(residuals, dtype=float), axis=0
                    ) / denom
                    measured.append({
                        "name": name,
                        "key": key,
                        # Keep the stable V3 logical ordering for continuity.
                        "station": float(data["station"]),
                        "projected_station": float(np.median(projected_stations))
                            if projected_stations else float(data["station"]),
                        "deviation": max(errors) if errors else 0.0,
                        "rms": float(np.sqrt(np.mean(np.asarray(errors) ** 2))) if errors else 0.0,
                        "pivot_z": pivot_z,
                        "top_zs": top_zs,
                        "correction_slope": correction_slope,
                    })
            return measured

        def trial_top_edge_move(top_array, bottom_array, item, fraction, max_shift):
            correction_slope = np.asarray(item["correction_slope"], dtype=float) * float(fraction)
            pivot_z = float(item["pivot_z"])
            top_zs = np.asarray(item["top_zs"], dtype=float)
            shifts = np.linalg.norm(
                (top_zs - pivot_z)[:, None] * correction_slope[None, :], axis=1
            )
            peak = float(np.max(shifts)) if len(shifts) else 0.0
            if max_shift > 0.0 and peak > max_shift:
                correction_slope *= max_shift / peak
                peak = max_shift
            key = item["key"]
            trial_top = np.asarray(top_array, dtype=float).copy()
            trial_bottom = np.asarray(bottom_array, dtype=float).copy()
            z1 = float(trial_top[key][2])
            z2 = float(trial_bottom[key][2])
            trial_top[key][:2] += correction_slope * (z1 - pivot_z)
            trial_bottom[key][:2] += correction_slope * (z2 - pivot_z)
            return trial_top, trial_bottom, peak

        # V3.9 performance/safety helper.  A single COORD-pillar move can
        # only change the cells that share that pillar.  Re-evaluating all
        # NX*NY*NZ cells for every candidate/line-search step made a complete
        # candidate scan unnecessarily expensive.  Keep the accepted global
        # signed Jacobians and replace only the affected local cells in trials.
        def affected_cells(key):
            pj, pi = key
            cells = []
            for cj in (pj - 1, pj):
                if cj < 0 or cj >= int(self.model.ny):
                    continue
                for ci in (pi - 1, pi):
                    if ci < 0 or ci >= int(self.model.nx):
                        continue
                    for ck in range(int(self.model.layers)):
                        cells.append((ck, cj, ci))
            return cells

        def cell_jacobian(top_array, bottom_array, k, j, i):
            keys = ((j, i), (j, i + 1), (j + 1, i), (j + 1, i + 1))
            top_vertices = []
            bottom_vertices = []
            for corner, pkey in enumerate(keys):
                pj, pi = pkey
                top_vertices.append(
                    self._point_on_coord_pillar(
                        top_array[pj, pi], bottom_array[pj, pi],
                        interfaces[k, j, i, corner],
                    )
                )
                bottom_vertices.append(
                    self._point_on_coord_pillar(
                        top_array[pj, pi], bottom_array[pj, pi],
                        interfaces[k + 1, j, i, corner],
                    )
                )
            v000, v100, v010, v110 = top_vertices
            v001, v101, v011, v111 = bottom_vertices
            di = ((v100 + v110 + v101 + v111) -
                  (v000 + v010 + v001 + v011)) / 4.0
            dj = ((v010 + v110 + v011 + v111) -
                  (v000 + v100 + v001 + v101)) / 4.0
            dk = ((v001 + v101 + v011 + v111) -
                  (v000 + v100 + v010 + v110)) / 4.0
            return float(np.linalg.det(np.stack((di, dj, dk), axis=1)))

        top_current_dets = self._cell_center_jacobians_from_coords(
            result_top, result_bottom, interfaces
        )
        top_current_signed = orientation * top_current_dets

        def top_trial_quality(top_array, bottom_array, key):
            trial_signed = top_current_signed.copy()
            updates = []
            for k, j, i in affected_cells(key):
                det = cell_jacobian(top_array, bottom_array, k, j, i)
                signed = orientation * det
                trial_signed[k, j, i] = signed
                updates.append((k, j, i, signed))
            finite_mask = np.isfinite(trial_signed)
            bad = int(np.count_nonzero(
                ~finite_mask
                | ((trial_signed <= 0.0) & ~pinched)
            ))
            finite_signed = trial_signed[finite_mask & ~pinched]
            min_ratio = (
                float(np.min(finite_signed)) / median_abs
                if len(finite_signed) else -np.inf
            )
            return bad, min_ratio, updates

        def accept_top_quality_updates(updates):
            for k, j, i, signed in updates:
                top_current_signed[k, j, i] = signed

        top_passes = max(0, int(getattr(
            self.model, "fault_coord_top_polish_passes", 6
        )))
        top_fraction = min(1.0, max(0.0, float(getattr(
            self.model, "fault_coord_top_polish_fraction", 0.80
        ))))
        top_tolerance = max(
            1.0e-9,
            float(getattr(
                self.model, "fault_coord_top_tolerance_fraction", 0.12
            )) * spacing,
        )
        top_max_shift = max(
            0.0,
            float(getattr(
                self.model, "fault_coord_top_polish_max_shift_fraction", 0.12
            )) * spacing,
        )

        def local_top_sumsq(measured, item, radius=1):
            """Mismatch energy for one pillar and its immediate fault neighbours."""
            same_fault = sorted(
                [x for x in measured if x["name"] == item["name"]],
                key=lambda x: float(x["station"]),
            )
            index = next(
                (idx for idx, x in enumerate(same_fault) if x["key"] == item["key"]),
                None,
            )
            if index is None:
                return float("inf")
            lo = max(0, index - int(radius))
            hi = min(len(same_fault), index + int(radius) + 1)
            return float(sum(x["deviation"] ** 2 for x in same_fault[lo:hi]))

        # V3.9: the V3.8 loop stopped as soon as the strongest remaining
        # upper-edge pillar hit the Jacobian floor.  That left visibly useful
        # movement unused at other fault stations.  V3.9 scans the complete
        # candidate list in descending mismatch order.  A rejected pillar is
        # recorded and skipped; the first safe candidate is accepted and the
        # geometry is then re-measured before the next cycle.
        #
        # The continuity guard is deliberately local: an accepted move must
        # reduce the mismatch energy of the candidate + its immediate along-
        # fault neighbours, while never increasing the global maximum mismatch.
        scan_all = bool(getattr(self.model, "fault_coord_top_scan_all", True))
        continuity_guard = bool(getattr(
            self.model, "fault_coord_top_continuity_guard", True
        ))

        for pass_number in range(1, top_passes + 1):
            before_all = measure_top_edge(result_top, result_bottom)
            candidates = [x for x in before_all if x["deviation"] > top_tolerance]
            if not candidates:
                break
            candidates.sort(key=lambda x: x["deviation"], reverse=True)
            before_max = max(x["deviation"] for x in before_all)
            before_sumsq = sum(x["deviation"] ** 2 for x in before_all)
            accepted_this_pass = False

            for candidate_rank, item in enumerate(candidates, start=1):
                local_before = local_top_sumsq(before_all, item)
                trial_top, trial_bottom, applied = trial_top_edge_move(
                    result_top, result_bottom, item, top_fraction, top_max_shift
                )
                bad, min_ratio, quality_updates = top_trial_quality(
                    trial_top, trial_bottom, item["key"]
                )
                after_all = measure_top_edge(trial_top, trial_bottom)
                after_lookup = {x["key"]: x for x in after_all}
                after_item = after_lookup.get(item["key"])
                deviation_after = (
                    float(after_item["deviation"])
                    if after_item is not None else float(item["deviation"])
                )
                after_max = max((x["deviation"] for x in after_all), default=0.0)
                after_sumsq = sum(x["deviation"] ** 2 for x in after_all)
                local_after = local_top_sumsq(after_all, item)

                # Unlike V3.8, a weaker candidate is allowed to move even when
                # the strongest (unchanged) pillar keeps the same global max.
                # It may never *increase* that max, and the total/local mismatch
                # energy must decrease.
                improved = (
                    deviation_after < item["deviation"] - 1.0e-9
                    and after_max <= before_max + 1.0e-9
                    and after_sumsq < before_sumsq - 1.0e-9
                    and (
                        not continuity_guard
                        or local_after < local_before - 1.0e-9
                    )
                )
                top_min_required = max(
                    min_required,
                    float(getattr(
                        self.model, "fault_coord_top_min_jacobian_ratio", 0.10
                    )),
                )
                accepted = bad == 0 and min_ratio >= top_min_required and improved
                adaptive_used = False
                requested_applied = float(applied)

                adaptive_enabled = bool(getattr(
                    self.model, "fault_coord_top_adaptive", True
                ))
                line_search_steps = max(1, int(getattr(
                    self.model, "fault_coord_top_line_search_steps", 12
                )))
                min_step_fraction = min(1.0, max(0.0, float(getattr(
                    self.model, "fault_coord_top_min_step_fraction", 0.02
                ))))

                if not accepted and adaptive_enabled and requested_applied > 1.0e-12:
                    low = 0.0
                    high = 1.0
                    best = None
                    for _ in range(line_search_steps):
                        alpha = 0.5 * (low + high)
                        cand_top, cand_bottom, cand_applied = trial_top_edge_move(
                            result_top, result_bottom, item,
                            top_fraction * alpha, top_max_shift * alpha
                        )
                        cand_bad, cand_ratio, cand_updates = top_trial_quality(
                            cand_top, cand_bottom, item["key"]
                        )
                        cand_all = measure_top_edge(cand_top, cand_bottom)
                        cand_lookup = {x["key"]: x for x in cand_all}
                        cand_item = cand_lookup.get(item["key"])
                        cand_dev = (
                            float(cand_item["deviation"])
                            if cand_item is not None else float(item["deviation"])
                        )
                        cand_max = max(
                            (x["deviation"] for x in cand_all), default=0.0
                        )
                        cand_sumsq = sum(x["deviation"] ** 2 for x in cand_all)
                        cand_local = local_top_sumsq(cand_all, item)
                        cand_improved = (
                            cand_dev < item["deviation"] - 1.0e-9
                            and cand_max <= before_max + 1.0e-9
                            and cand_sumsq < before_sumsq - 1.0e-9
                            and (
                                not continuity_guard
                                or cand_local < local_before - 1.0e-9
                            )
                        )
                        cand_ok = (
                            cand_bad == 0
                            and cand_ratio >= top_min_required
                            and cand_improved
                        )
                        if cand_ok:
                            best = (
                                alpha, cand_top, cand_bottom, float(cand_applied),
                                float(cand_ratio), float(cand_dev), float(cand_max),
                                float(cand_sumsq), float(cand_local), cand_updates
                            )
                            low = alpha
                        else:
                            high = alpha

                    if best is not None and best[0] >= min_step_fraction:
                        (
                            _alpha, trial_top, trial_bottom, applied, min_ratio,
                            deviation_after, after_max, after_sumsq, local_after,
                            quality_updates
                        ) = best
                        accepted = True
                        adaptive_used = True

                if accepted:
                    result_top = trial_top
                    result_bottom = trial_bottom
                    accept_top_quality_updates(quality_updates)
                else:
                    applied = 0.0
                    deviation_after = item["deviation"]

                records.append({
                    "phase": "TOP_POLISH",
                    "pass": pass_number,
                    "candidate_rank": int(candidate_rank),
                    "name": item["name"],
                    "key": item["key"],
                    "station": item["station"],
                    "deviation_before": item["deviation"],
                    "deviation_after": deviation_after,
                    "applied_shift": float(applied),
                    "requested_shift": float(requested_applied),
                    "adaptive": bool(adaptive_used),
                    "accepted": accepted,
                    "trial_min_jacobian_ratio": min_ratio,
                    "max_deviation_before": before_max,
                    "max_deviation_after": after_max if accepted else before_max,
                    "sumsq_before": before_sumsq,
                    "sumsq_after": after_sumsq if accepted else before_sumsq,
                    "local_sumsq_before": local_before,
                    "local_sumsq_after": local_after if accepted else local_before,
                })

                if accepted:
                    accepted_this_pass = True
                    # Re-measure after every accepted move.  The remaining
                    # candidate order and safe Jacobian room can both change.
                    break
                if not scan_all:
                    break

            if not accepted_this_pass:
                # Every remaining candidate has been tried and none can move
                # safely/improvingly.  Only now is polishing finished.
                break

        final_top_measured = measure_top_edge(result_top, result_bottom)
        final_top_max = max((x["deviation"] for x in final_top_measured), default=0.0)
        final_top_over = sum(x["deviation"] > top_tolerance for x in final_top_measured)

        final_measured = measure(result_top, result_bottom)
        final_max = max((x["deviation"] for x in final_measured), default=0.0)
        final_over = sum(x["deviation"] > tolerance for x in final_measured)
        self._coord_alignment_records = records
        self._coord_alignment_summary = {
            "checked": sum(len(v) for v in anchors_by_name.values()),
            "tolerance": tolerance,
            "primary_detected": len(primary_candidates),
            "primary_accepted": sum(
                1 for r in records
                if r["phase"] == "PRIMARY" and r["accepted"]
            ),
            "polish_attempted": sum(1 for r in records if r["phase"] == "POLISH"),
            "polish_accepted": sum(
                1 for r in records
                if r["phase"] == "POLISH" and r["accepted"]
            ),
            "top_polish_attempted": sum(1 for r in records if r["phase"] == "TOP_POLISH"),
            "top_polish_accepted": sum(
                1 for r in records
                if r["phase"] == "TOP_POLISH" and r["accepted"]
            ),
            "remaining_above_tolerance": int(final_over),
            "final_max_deviation": float(final_max),
            "top_remaining_above_tolerance": int(final_top_over),
            "top_final_max_deviation": float(final_top_max),
            "top_tolerance": float(top_tolerance),
        }

        primary_accepted = self._coord_alignment_summary["primary_accepted"]
        polish_accepted = self._coord_alignment_summary["polish_accepted"]
        top_polish_accepted = self._coord_alignment_summary["top_polish_accepted"]
        if not primary_candidates and polish_accepted == 0:
            print(
                "PyGRID: COORD fault-face alignment: all fault-edge sticks "
                f"within {tolerance:.2f} map units; no correction required."
            )
        else:
            print(
                "PyGRID: COORD fault-face alignment: "
                f"{len(primary_candidates)} primary outlier(s), "
                f"{primary_accepted} corrected; "
                f"{polish_accepted} ribbon polish move(s), "
                f"{top_polish_accepted} upper-edge polish move(s) accepted. "
                f"Final ribbon mismatch = {final_max:.2f}; "
                f"upper-edge mismatch = {final_top_max:.2f}."
            )
        return result_top, result_bottom

    def _conform_fault_trace_band(self, coord_top, coord_bottom):
        """Move the actual TOP fault edge toward FLT with a tapered grid band.

        V3.10-V3.21 attempted to correct the visible top fault line by rotating
        one COORD stick at a time about its lower fault-edge pivot.  On coarse
        grids the required move can make one of the four cells sharing that
        pillar fail the Jacobian guard, so the move is rejected even though the
        *same* geometric correction is safe when the surrounding grid moves
        with it.

        V3.22 therefore treats FLT as the authoritative top-fault trace and
        applies every anchor correction as a small deformation field.  The
        fault-edge pillar receives 100% of the requested correction and the
        existing transverse logical line receives a linear taper over the next
        ``fault_trace_conform_lines`` pillars (default four: 80/60/40/20%).
        Where deformation bands overlap, non-anchor moves are blended.  All
        anchor and neighbour moves are then accepted collectively through one
        global Jacobian line search.

        ZCORN and logical FAULTS topology are unchanged.  Only COORD XY is
        changed, so the fault may cross I and J faces without the visible green
        edge being forced to kink with every logical row/column switch.
        """
        self._fault_trace_conforming_summary = {}
        if not bool(getattr(self.model, "fault_trace_conforming", True)):
            return coord_top, coord_bottom
        if not (
            bool(getattr(self.model, "conform_slanted_faults", False))
            and self._slanted_fault_mode() in {"COORD", "MIXED"}
            and self.model.split_faults
            and self.model.fault_set
            and self.model.fault_set.slanted_names
        ):
            return coord_top, coord_bottom

        top_in = np.asarray(coord_top, dtype=float)
        bottom_in = np.asarray(coord_bottom, dtype=float)
        interfaces = self._cell_corner_interfaces()
        nz = int(self.model.layers)

        base = self._base_pillar_array()
        anchors_by_name = self._fault_chain_anchors(base)
        if not anchors_by_name:
            return coord_top, coord_bottom

        # Actual top/bottom fault-edge Z values belonging to each logical
        # fault pillar.  These are the depths ResInsight uses to draw the
        # visible fault faces, so alignment must be measured here rather than
        # only at the structural fault-plane reference depth.
        face_depths = {}
        for face_name, i1, j1, face in self._raw_fault_faces():
            i0 = i1 - 1
            j0 = j1 - 1
            if face == "X+":
                endpoints = (((j0, i0 + 1), 1), ((j0 + 1, i0 + 1), 3))
            elif face == "Y+":
                endpoints = (((j0 + 1, i0), 2), ((j0 + 1, i0 + 1), 3))
            else:
                continue
            for key, corner in endpoints:
                bucket = face_depths.setdefault((face_name, key), {
                    "top": [], "bottom": []
                })
                bucket["top"].append(float(interfaces[0, j0, i0, corner]))
                bucket["bottom"].append(float(interfaces[nz, j0, i0, corner]))

        # Reuse the transverse family chosen when the TOP grid was conformed
        # to FLT.  It is deterministic at I/J switches and represents the grid
        # line that should absorb the anchor displacement.
        axis_by_key = {}
        for item in getattr(self, "_conforming_transverse_segments", ()):
            key = tuple(item.get("key", ()))
            axis = item.get("axis")
            if len(key) == 2 and axis in {"I", "J"}:
                axis_by_key[key] = axis

        ny, nx = int(self.model.ny), int(self.model.nx)
        lines = max(1, int(getattr(self.model, "fault_trace_conform_lines", 4)))

        # Endpoint deltas for non-anchor band pillars.  Do not normalise a lone
        # taper weight back to one; a single offset-1 pillar must really receive
        # 80% rather than 100% of the anchor correction.
        top_num = np.zeros_like(top_in[..., :2])
        bottom_num = np.zeros_like(bottom_in[..., :2])
        weights = np.zeros((ny + 1, nx + 1), dtype=float)
        anchor_target_top = {}
        anchor_target_bottom = {}
        records = []

        bottom_by_top_id = {}
        for name in self.model.fault_set.slanted_names:
            for top_trace, bottom_trace in self.model.fault_set.matched_trace_pairs(name):
                bottom_by_top_id[id(top_trace)] = bottom_trace

        for name, anchors in anchors_by_name.items():
            ordered = sorted(
                anchors.items(), key=lambda item: float(item[1]["station"])
            )
            for key, data in ordered:
                depth_data = face_depths.get((name, key))
                top_trace = data.get("trace")
                bottom_trace = bottom_by_top_id.get(id(top_trace))
                axis = axis_by_key.get(key)
                if (
                    top_trace is None
                    or bottom_trace is None
                    or axis not in {"I", "J"}
                    or not depth_data
                    or not depth_data["top"]
                    or not depth_data["bottom"]
                ):
                    continue

                top_zs = np.unique(np.asarray(depth_data["top"], dtype=float))
                bottom_zs = np.unique(np.asarray(depth_data["bottom"], dtype=float))
                pivot_z = float(np.median(bottom_zs))

                # Same FLT correction requested by the old TOP_POLISH stage,
                # but applied with the surrounding grid instead of to one
                # isolated stick.  Rotating about the lower edge gives the
                # strongest visible top-line improvement for a given move.
                residuals = []
                dzs = []
                before_errors = []
                for z in top_zs:
                    current_xy = self._point_on_coord_pillar(
                        top_in[key], bottom_in[key], float(z)
                    )[:2]
                    hit = _project_point_to_trace(current_xy, top_trace)
                    desired_xy = np.asarray(hit[0], dtype=float)
                    residuals.append(desired_xy - current_xy)
                    dzs.append(float(z) - pivot_z)
                    before_errors.append(float(hit[1]))

                denom = float(np.sum(np.asarray(dzs, dtype=float) ** 2))
                if denom <= 1.0e-12:
                    continue
                correction_slope = np.sum(
                    np.asarray(dzs, dtype=float)[:, None]
                    * np.asarray(residuals, dtype=float),
                    axis=0,
                ) / denom
                if float(np.linalg.norm(correction_slope)) <= 1.0e-14:
                    continue

                z1 = float(top_in[key][2])
                z2 = float(bottom_in[key][2])
                own_top_delta = correction_slope * (z1 - pivot_z)
                own_bottom_delta = correction_slope * (z2 - pivot_z)
                anchor_target_top[key] = top_in[key][:2] + own_top_delta
                anchor_target_bottom[key] = bottom_in[key][:2] + own_bottom_delta

                # Keep the interpreted FLT->FLB generator information for the
                # collective line search.  V3.22 moves as far toward FLT as
                # possible without sacrificing the v3.20 requirement that the
                # median retained fault slant stays at least 75% by default.
                station = float(data["station"])
                top_target = np.asarray(data["target"], dtype=float)
                lower_target = _bottom_point_for_top_station(
                    top_trace, bottom_trace, station
                )
                retention_data = None
                if lower_target is not None:
                    lower_target = np.asarray(lower_target, dtype=float)
                    plane_depths = self._fault_reference_depths(
                        name, top_target, lower_target
                    )
                    if plane_depths is not None:
                        rz_top, rz_bottom = map(float, plane_depths)
                        ideal_vector = lower_target - top_target
                        ideal_norm2 = float(np.dot(ideal_vector, ideal_vector))
                        if ideal_norm2 > 1.0e-18 and abs(rz_bottom-rz_top) > 1.0e-12:
                            retention_data = (
                                top_target, lower_target, rz_top, rz_bottom,
                                ideal_vector, ideal_norm2,
                            )

                j, i = key
                for offset in range(-lines, lines + 1):
                    if offset == 0:
                        continue
                    if axis == "I":
                        jj, ii = j, i + offset
                    else:
                        jj, ii = j + offset, i
                    if not (0 <= jj <= ny and 0 <= ii <= nx):
                        continue

                    taper = max(
                        0.0,
                        (lines + 1 - abs(offset)) / float(lines + 1),
                    )
                    if taper <= 1.0e-12:
                        continue
                    nz1 = float(top_in[jj, ii, 2])
                    nz2 = float(bottom_in[jj, ii, 2])
                    top_num[jj, ii] += taper * correction_slope * (nz1 - pivot_z)
                    bottom_num[jj, ii] += taper * correction_slope * (nz2 - pivot_z)
                    weights[jj, ii] += taper

                records.append({
                    "name": name,
                    "key": key,
                    "station": station,
                    "axis": axis,
                    "before_max": max(before_errors) if before_errors else 0.0,
                    "retention_data": retention_data,
                })

        if not records:
            return coord_top, coord_bottom

        target_top = top_in.copy()
        target_bottom = bottom_in.copy()
        anchor_keys = set(anchor_target_top)

        mask = weights > 1.0e-12
        for j, i in zip(*np.nonzero(mask)):
            key = (int(j), int(i))
            if key in anchor_keys:
                continue
            denom = max(1.0, float(weights[j, i]))
            target_top[j, i, :2] = top_in[j, i, :2] + top_num[j, i] / denom
            target_bottom[j, i, :2] = (
                bottom_in[j, i, :2] + bottom_num[j, i] / denom
            )

        # Fault anchors are authoritative: their own FLT correction is not
        # diluted by overlapping neighbouring bands.
        for key in anchor_keys:
            target_top[key][:2] = anchor_target_top[key]
            target_bottom[key][:2] = anchor_target_bottom[key]

        pinched = self._pinched_cell_mask(interfaces)
        baseline_dets = self._cell_center_jacobians_from_coords(
            top_in, bottom_in, interfaces
        )
        finite = baseline_dets[np.isfinite(baseline_dets) & ~pinched]
        orientation = (
            1.0 if len(finite) == 0 or float(np.median(finite)) >= 0.0 else -1.0
        )
        median_abs = max(
            abs(float(np.median(finite))) if len(finite) else 1.0, 1.0e-30
        )
        min_required = max(0.0, float(getattr(
            self.model, "fault_trace_conform_min_jacobian_ratio", 0.02
        )))

        min_median_slant = min(1.0, max(0.0, float(getattr(
            self.model, "fault_trace_conform_min_median_slant_fraction", 0.75
        ))))

        def median_retained_slant(top_array, bottom_array):
            values = []
            for item in records:
                data = item.get("retention_data")
                if data is None:
                    continue
                (
                    top_target, _lower_target, rz_top, rz_bottom,
                    ideal_vector, ideal_norm2,
                ) = data
                key = item["key"]
                actual_top = self._point_on_coord_pillar(
                    top_array[key], bottom_array[key], rz_top
                )[:2]
                actual_bottom = self._point_on_coord_pillar(
                    top_array[key], bottom_array[key], rz_bottom
                )[:2]
                retention = float(
                    np.dot(actual_bottom - actual_top, ideal_vector) / ideal_norm2
                )
                values.append(retention)
            return float(np.median(values)) if values else 1.0

        def quality(alpha):
            alpha = min(1.0, max(0.0, float(alpha)))
            trial_top = top_in.copy()
            trial_bottom = bottom_in.copy()
            trial_top[..., :2] = (
                top_in[..., :2]
                + alpha * (target_top[..., :2] - top_in[..., :2])
            )
            trial_bottom[..., :2] = (
                bottom_in[..., :2]
                + alpha * (target_bottom[..., :2] - bottom_in[..., :2])
            )
            dets = self._cell_center_jacobians_from_coords(
                trial_top, trial_bottom, interfaces
            )
            signed = orientation * dets
            bad = int(np.count_nonzero(
                ~np.isfinite(signed) | ((signed <= 0.0) & ~pinched)
            ))
            active = signed[np.isfinite(signed) & ~pinched]
            ratio = (
                float(np.min(active)) / median_abs if len(active) else -np.inf
            )
            median_slant = median_retained_slant(trial_top, trial_bottom)
            return trial_top, trial_bottom, bad, ratio, median_slant

        result_top, result_bottom, bad, min_ratio, median_slant = quality(1.0)
        alpha = 1.0
        if (
            bad != 0
            or min_ratio < min_required
            or median_slant < min_median_slant
        ):
            low = 0.0
            high = 1.0
            base_top, base_bottom, base_bad, base_ratio, base_slant = quality(0.0)
            best = (base_top, base_bottom, 0.0, base_ratio, base_slant)
            steps = max(1, int(getattr(
                self.model, "fault_trace_conform_line_search_steps", 14
            )))
            for _ in range(steps):
                mid = 0.5 * (low + high)
                (
                    cand_top, cand_bottom, cand_bad, cand_ratio, cand_slant
                ) = quality(mid)
                if (
                    cand_bad == 0
                    and cand_ratio >= min_required
                    and cand_slant >= min_median_slant
                ):
                    best = (cand_top, cand_bottom, mid, cand_ratio, cand_slant)
                    low = mid
                else:
                    high = mid
            result_top, result_bottom, alpha, min_ratio, median_slant = best

        # Measure the actual visible top fault line after the collective move.
        after_errors = []
        before_errors = []
        for item in records:
            name = item["name"]
            key = item["key"]
            data = anchors_by_name[name][key]
            trace = data.get("trace")
            depth_data = face_depths.get((name, key), {})
            for z in np.unique(np.asarray(depth_data.get("top", ()), dtype=float)):
                before_xy = self._point_on_coord_pillar(
                    top_in[key], bottom_in[key], float(z)
                )[:2]
                after_xy = self._point_on_coord_pillar(
                    result_top[key], result_bottom[key], float(z)
                )[:2]
                before_errors.append(float(_project_point_to_trace(before_xy, trace)[1]))
                after_errors.append(float(_project_point_to_trace(after_xy, trace)[1]))

        before_max = max(before_errors) if before_errors else 0.0
        after_max = max(after_errors) if after_errors else 0.0
        before_rms = (
            float(np.sqrt(np.mean(np.asarray(before_errors) ** 2)))
            if before_errors else 0.0
        )
        after_rms = (
            float(np.sqrt(np.mean(np.asarray(after_errors) ** 2)))
            if after_errors else 0.0
        )
        moved_band = int(np.count_nonzero(
            np.linalg.norm(
                result_top[..., :2] - top_in[..., :2], axis=2
            ) > 1.0e-9
        ))
        self._fault_trace_conforming_summary = {
            "anchors": len(anchor_keys),
            "band_pillars": moved_band,
            "lines": lines,
            "accepted_fraction": float(alpha),
            "before_max": float(before_max),
            "after_max": float(after_max),
            "before_rms": float(before_rms),
            "after_rms": float(after_rms),
            "min_jacobian_ratio": float(min_ratio),
            "median_retained_slant": float(median_slant),
        }
        print(
            "PyGRID: FLT-trace conforming: "
            f"{len(anchor_keys)} anchors with {lines}-line tapered bands; "
            f"collective move fraction = {alpha:.3f}; "
            f"top-edge max mismatch {before_max:.2f} -> {after_max:.2f}; "
            f"RMS {before_rms:.2f} -> {after_rms:.2f}; "
            f"minimum centre-Jacobian ratio = {min_ratio:.4f}; "
            f"median retained slant = {median_slant:.3f}."
        )
        return result_top, result_bottom

    def _redistribute_fault_trace_band_harmonic(
        self, seed_top, seed_bottom, final_top, final_bottom
    ):
        """V4.1: smooth only the *surrounding* accepted fault deformation.

        The complete v3.22/V4.0 fault pipeline is run first, including FLT-trace
        conforming and fault-plane-generator straightening.  Those final fault
        pillars are then frozen.  V4.1 redistributes only the displacement of
        nearby non-fault pillars by solving a local discrete harmonic field.

        This ordering is deliberate: V4.1 can visibly improve the shape of the
        cells around a fault but cannot move the already accepted fault edge,
        change the logical FAULTS path, change NX/NY, or alter the model
        footprint.  A 3-D Jacobian line search rejects any unsafe redistribution.
        """
        enabled = bool(getattr(self.model, "fault_band_relaxation", False))
        self._fault_band_relaxation_summary = {
            "enabled": enabled,
            "accepted_fraction": 0.0,
            "moved_pillars": 0,
            "before_roughness": 0.0,
            "after_roughness": 0.0,
            "min_jacobian_ratio": 0.0,
        }
        if not enabled:
            return final_top, final_bottom
        if not (
            bool(getattr(self.model, "conform_slanted_faults", False))
            and self.model.split_faults
            and self.model.fault_set
        ):
            return final_top, final_bottom

        seed_top = np.asarray(seed_top, dtype=float)
        seed_bottom = np.asarray(seed_bottom, dtype=float)
        final_top = np.asarray(final_top, dtype=float)
        final_bottom = np.asarray(final_bottom, dtype=float)
        ny, nx = self.model.ny, self.model.nx

        base = self._base_pillar_array()
        anchors_by_name = self._fault_chain_anchors(
            base, names=self._conforming_fault_names()
        )
        anchor_keys = {
            tuple(key)
            for anchors in anchors_by_name.values()
            for key in anchors.keys()
        }
        if not anchor_keys:
            return final_top, final_bottom

        lines = max(1, int(getattr(
            self.model, "fault_band_relax_lines", 6
        )))
        iterations = max(1, int(getattr(
            self.model, "fault_band_relax_iterations", 80
        )))
        strength = min(1.0, max(0.0, float(getattr(
            self.model, "fault_band_relax_strength", 0.85
        ))))
        blend = min(1.0, max(0.0, float(getattr(
            self.model, "fault_band_relax_blend", 0.75
        ))))

        core = np.zeros((ny + 1, nx + 1), dtype=bool)
        ring = np.zeros_like(core)
        for j0, i0 in anchor_keys:
            for radius, target_mask in ((lines, core), (lines + 1, ring)):
                for dj in range(-radius, radius + 1):
                    remain = radius - abs(dj)
                    j = j0 + dj
                    if not (0 <= j <= ny):
                        continue
                    ilo = max(0, i0 - remain)
                    ihi = min(nx, i0 + remain)
                    target_mask[j, ilo:ihi + 1] = True

        anchor_mask = np.zeros_like(core)
        for j, i in anchor_keys:
            if 0 <= j <= ny and 0 <= i <= nx:
                anchor_mask[j, i] = True
        outer = np.zeros_like(core)
        outer[0, :] = True
        outer[ny, :] = True
        outer[:, 0] = True
        outer[:, nx] = True
        movable = core & ~anchor_mask & ~outer
        if not np.any(movable):
            return final_top, final_bottom

        seed_disp_top = final_top[..., :2] - seed_top[..., :2]
        seed_disp_bottom = final_bottom[..., :2] - seed_bottom[..., :2]

        def solve(seed_disp):
            # Start from the already accepted displacement field.  Exact fault
            # pillars, the one-cell outer ring of the local band, and the model
            # boundary are Dirichlet controls.  Only the interior non-fault
            # pillars are allowed to relax.
            field = seed_disp.copy()
            ring_boundary = ring & ~core
            fixed = anchor_mask | ring_boundary | outer | ~ring
            active = core & ~fixed
            for _ in range(iterations):
                old = field.copy()
                total = np.zeros_like(field)
                count = np.zeros((ny + 1, nx + 1), dtype=float)
                total[1:, :] += old[:-1, :]
                count[1:, :] += 1.0
                total[:-1, :] += old[1:, :]
                count[:-1, :] += 1.0
                total[:, 1:] += old[:, :-1]
                count[:, 1:] += 1.0
                total[:, :-1] += old[:, 1:]
                count[:, :-1] += 1.0
                harmonic = total / np.maximum(count[..., None], 1.0)
                field[active] = (
                    (1.0 - strength) * old[active]
                    + strength * harmonic[active]
                )
                # Reimpose the fixed accepted geometry exactly.
                field[fixed] = seed_disp[fixed]

            target = seed_disp.copy()
            target[movable] = (
                (1.0 - blend) * seed_disp[movable]
                + blend * field[movable]
            )
            target[anchor_mask] = seed_disp[anchor_mask]
            target[outer & ~anchor_mask] = seed_disp[outer & ~anchor_mask]
            return target

        target_disp_top = solve(seed_disp_top)
        target_disp_bottom = solve(seed_disp_bottom)

        def roughness(top_disp, bottom_disp):
            value = 0.0
            count = 0
            for disp in (top_disp, bottom_disp):
                di = disp[:, 1:] - disp[:, :-1]
                dj = disp[1:, :] - disp[:-1, :]
                mask_i = core[:, 1:] | core[:, :-1]
                mask_j = core[1:, :] | core[:-1, :]
                if np.any(mask_i):
                    value += float(np.sum(di[mask_i] ** 2))
                    count += int(np.count_nonzero(mask_i))
                if np.any(mask_j):
                    value += float(np.sum(dj[mask_j] ** 2))
                    count += int(np.count_nonzero(mask_j))
            return value / max(count, 1)

        before_rough = roughness(seed_disp_top, seed_disp_bottom)
        target_rough = roughness(target_disp_top, target_disp_bottom)
        if not np.isfinite(target_rough) or target_rough >= before_rough - 1.0e-12:
            self._fault_band_relaxation_summary.update({
                "before_roughness": float(before_rough),
                "after_roughness": float(before_rough),
            })
            return final_top, final_bottom

        interfaces = self._cell_corner_interfaces()
        pinched = self._pinched_cell_mask(interfaces)
        base_dets = self._cell_center_jacobians_from_coords(
            final_top, final_bottom, interfaces
        )
        finite = base_dets[np.isfinite(base_dets) & ~pinched]
        orientation = (
            1.0 if len(finite) == 0 or float(np.median(finite)) >= 0.0 else -1.0
        )
        median_abs = max(
            abs(float(np.median(finite))) if len(finite) else 1.0, 1.0e-30
        )
        signed = orientation * base_dets
        active = signed[np.isfinite(signed) & ~pinched]
        base_ratio = float(np.min(active)) / median_abs if len(active) else -np.inf
        absolute_floor = max(0.0, float(getattr(
            self.model, "fault_trace_conform_min_jacobian_ratio", 0.02
        )))
        required_ratio = max(absolute_floor, 0.95 * base_ratio)

        def quality(alpha):
            alpha = min(1.0, max(0.0, float(alpha)))
            top = final_top.copy()
            bottom = final_bottom.copy()
            top[..., :2] = (
                final_top[..., :2]
                + alpha * (
                    (seed_top[..., :2] + target_disp_top)
                    - final_top[..., :2]
                )
            )
            bottom[..., :2] = (
                final_bottom[..., :2]
                + alpha * (
                    (seed_bottom[..., :2] + target_disp_bottom)
                    - final_bottom[..., :2]
                )
            )
            # The final v3.22/V4.0 fault edge is immutable.
            top[anchor_mask] = final_top[anchor_mask]
            bottom[anchor_mask] = final_bottom[anchor_mask]
            top[outer & ~anchor_mask] = final_top[outer & ~anchor_mask]
            bottom[outer & ~anchor_mask] = final_bottom[outer & ~anchor_mask]

            dets = self._cell_center_jacobians_from_coords(top, bottom, interfaces)
            sgn = orientation * dets
            bad = int(np.count_nonzero(
                ~np.isfinite(sgn) | ((sgn <= 0.0) & ~pinched)
            ))
            vals = sgn[np.isfinite(sgn) & ~pinched]
            ratio = float(np.min(vals)) / median_abs if len(vals) else -np.inf
            return top, bottom, bad, ratio

        cand_top, cand_bottom, bad, ratio = quality(1.0)
        alpha = 1.0
        if bad or ratio < required_ratio:
            low, high = 0.0, 1.0
            base_top, base_bottom, _bad0, ratio0 = quality(0.0)
            best = (base_top, base_bottom, 0.0, ratio0)
            steps = max(8, int(getattr(
                self.model, "fault_trace_conform_line_search_steps", 14
            )))
            for _ in range(steps):
                mid = 0.5 * (low + high)
                t, b, bd, r = quality(mid)
                if bd == 0 and r >= required_ratio:
                    best = (t, b, mid, r)
                    low = mid
                else:
                    high = mid
            cand_top, cand_bottom, alpha, ratio = best

        if alpha <= 1.0e-6:
            self._fault_band_relaxation_summary.update({
                "before_roughness": float(before_rough),
                "after_roughness": float(before_rough),
                "min_jacobian_ratio": float(base_ratio),
                "lines": int(lines),
            })
            return final_top, final_bottom

        final_disp_top = cand_top[..., :2] - seed_top[..., :2]
        final_disp_bottom = cand_bottom[..., :2] - seed_bottom[..., :2]
        after_rough = roughness(final_disp_top, final_disp_bottom)
        move = np.maximum(
            np.linalg.norm(cand_top[..., :2] - final_top[..., :2], axis=2),
            np.linalg.norm(cand_bottom[..., :2] - final_bottom[..., :2], axis=2),
        )
        moved = int(np.count_nonzero(move > 1.0e-9))
        self._fault_band_relaxation_summary.update({
            "accepted_fraction": float(alpha),
            "moved_pillars": moved,
            "before_roughness": float(before_rough),
            "after_roughness": float(after_rough),
            "min_jacobian_ratio": float(ratio),
            "lines": int(lines),
        })
        print(
            "PyGRID V4.1: harmonic fault-band redistribution: "
            f"{moved} non-fault pillar(s) adjusted; "
            f"accepted fraction = {alpha:.3f}; "
            f"band roughness {before_rough:.3g} -> {after_rough:.3g}; "
            f"minimum centre-Jacobian ratio = {ratio:.4f}."
        )
        return cand_top, cand_bottom

    def _straighten_fault_plane_generators(self, coord_top, coord_bottom):
        """Regularise the final fault-edge COORD sticks along the ruled plane.

        V3.20 preserves the interpreted FLT->FLB slant with an adaptive 3-D
        topology guard.  On a coarse grid that guard can legitimately reduce
        different fault pillars by different amounts.  Later ribbon polishing
        may also rotate one isolated pillar slightly past the interpreted lower
        trace.  Both effects are safe numerically, but the visible fault plane
        can still acquire a kink where the logical path switches between I and
        J faces.

        V3.21 treats the interpreted ruled fault plane as the geometric
        reference after all existing safety/alignment passes:

        * each fault-edge generator is constrained to the FLT->FLB direction at
          its continuous fault station;
        * a generator is never moved *farther* toward FLB than the currently
          safe geometry permits;
        * the retained slant fraction is made Lipschitz-continuous along the
          fault chain by reducing only locally excessive neighbours;
        * the complete 3-D Jacobian QC is checked before the collective move is
          accepted, with a global line search fallback.

        Thus logical row/column switching remains an Eclipse topology detail
        rather than a geometric kink in the interpreted fault plane.
        """
        if not bool(getattr(self.model, "fault_plane_straightening", True)):
            self._fault_plane_straightening_summary = {}
            return coord_top, coord_bottom
        if not (
            bool(getattr(self.model, "conform_slanted_faults", False))
            and self._slanted_fault_mode() in {"COORD", "MIXED"}
            and self.model.split_faults
            and self.model.fault_set
            and self.model.fault_set.slanted_names
        ):
            self._fault_plane_straightening_summary = {}
            return coord_top, coord_bottom

        top_in = np.asarray(coord_top, dtype=float)
        bottom_in = np.asarray(coord_bottom, dtype=float)
        target_top = top_in.copy()
        target_bottom = bottom_in.copy()

        base = self._base_pillar_array()
        anchors_by_name = self._fault_chain_anchors(base)

        # Match each actual TOP trace object to its one-to-one FLB partner.
        bottom_by_top_id = {}
        for name in self.model.fault_set.slanted_names:
            for top_trace, bottom_trace in self.model.fault_set.matched_trace_pairs(name):
                bottom_by_top_id[id(top_trace)] = bottom_trace

        max_step = float(getattr(
            self.model, "fault_plane_max_retention_step", 0.15
        ))
        if max_step <= 0.0:
            max_step = 0.15

        prepared = []
        for name, anchors in anchors_by_name.items():
            ordered = sorted(
                anchors.items(), key=lambda item: float(item[1]["station"])
            )
            chain_items = []
            for key, data in ordered:
                top_trace = data.get("trace")
                bottom_trace = bottom_by_top_id.get(id(top_trace))
                if bottom_trace is None:
                    continue

                station = float(data["station"])
                top_target = np.asarray(data["target"], dtype=float)
                lower_target = _bottom_point_for_top_station(
                    top_trace, bottom_trace, station
                )
                if lower_target is None:
                    continue
                lower_target = np.asarray(lower_target, dtype=float)
                ideal_vector = lower_target - top_target
                ideal_norm2 = float(np.dot(ideal_vector, ideal_vector))
                if ideal_norm2 <= 1.0e-18:
                    continue

                depths = self._fault_reference_depths(
                    name, top_target, lower_target
                )
                if depths is None:
                    continue
                z_top, z_bottom = map(float, depths)
                dz = z_bottom - z_top
                if abs(dz) <= 1.0e-12:
                    continue

                actual_top = self._point_on_coord_pillar(
                    top_in[key], bottom_in[key], z_top
                )[:2]
                actual_bottom = self._point_on_coord_pillar(
                    top_in[key], bottom_in[key], z_bottom
                )[:2]
                current_vector = actual_bottom - actual_top

                # Scalar retention along the interpreted generator.  Clamp to
                # [0,1]: V3.21 may reduce a safe slant but never overshoots FLB
                # and never increases a topology-limited pillar's displacement.
                retention = float(
                    np.dot(current_vector, ideal_vector) / ideal_norm2
                )
                retention = min(1.0, max(0.0, retention))
                chain_items.append({
                    "key": key,
                    "station": station,
                    "top_target": top_target,
                    "lower_target": lower_target,
                    "ideal_vector": ideal_vector,
                    "z_top": z_top,
                    "z_bottom": z_bottom,
                    "retention": retention,
                })

            if len(chain_items) < 2:
                continue

            # Lower-envelope regularisation.  Because item i itself is included
            # in the minimum, the smoothed value can never exceed its current
            # safe retention.  A low topology-limited pillar therefore creates
            # only a gradual ramp in its neighbours rather than a sharp kink.
            current = np.asarray(
                [item["retention"] for item in chain_items], dtype=float
            )
            smooth = np.empty_like(current)
            for i in range(len(current)):
                smooth[i] = min(
                    current[j] + max_step * abs(i - j)
                    for j in range(len(current))
                )
            smooth = np.minimum(current, np.clip(smooth, 0.0, 1.0))

            for item, retention in zip(chain_items, smooth):
                key = item["key"]
                z_top = item["z_top"]
                z_bottom = item["z_bottom"]
                dz = z_bottom - z_top
                slope = retention * item["ideal_vector"] / dz
                z1 = float(target_top[key][2])
                z2 = float(target_bottom[key][2])
                target_top[key][:2] = (
                    item["top_target"] + slope * (z1 - z_top)
                )
                target_bottom[key][:2] = (
                    item["top_target"] + slope * (z2 - z_top)
                )
                prepared.append({
                    **item,
                    "target_retention": float(retention),
                })

        if not prepared:
            self._fault_plane_straightening_summary = {}
            return coord_top, coord_bottom

        interfaces = self._cell_corner_interfaces()
        pinched = self._pinched_cell_mask(interfaces)
        baseline_dets = self._cell_center_jacobians_from_coords(
            top_in, bottom_in, interfaces
        )
        finite = baseline_dets[np.isfinite(baseline_dets) & ~pinched]
        orientation = 1.0 if (len(finite) == 0 or float(np.median(finite)) >= 0.0) else -1.0
        median_abs = max(abs(float(np.median(finite))) if len(finite) else 1.0, 1.0e-30)
        min_required = max(0.0, float(getattr(
            self.model, "fault_plane_straightening_min_jacobian_ratio", 0.02
        )))

        moved_keys = {item["key"] for item in prepared}

        def trial(alpha):
            alpha = min(1.0, max(0.0, float(alpha)))
            trial_top = top_in.copy()
            trial_bottom = bottom_in.copy()
            for key in moved_keys:
                trial_top[key][:2] = (
                    top_in[key][:2]
                    + alpha * (target_top[key][:2] - top_in[key][:2])
                )
                trial_bottom[key][:2] = (
                    bottom_in[key][:2]
                    + alpha * (target_bottom[key][:2] - bottom_in[key][:2])
                )
            dets = self._cell_center_jacobians_from_coords(
                trial_top, trial_bottom, interfaces
            )
            signed = orientation * dets
            active = signed[np.isfinite(signed) & ~pinched]
            bad = int(np.count_nonzero(
                ~np.isfinite(signed) | ((signed <= 0.0) & ~pinched)
            ))
            ratio = (
                float(np.min(active)) / median_abs if len(active) else -np.inf
            )
            return trial_top, trial_bottom, bad, ratio

        result_top, result_bottom, bad, min_ratio = trial(1.0)
        alpha = 1.0
        if bad != 0 or min_ratio < min_required:
            low = 0.0
            high = 1.0
            best = (top_in.copy(), bottom_in.copy(), 0.0, 0.0)
            steps = max(1, int(getattr(
                self.model, "fault_plane_straightening_line_search_steps", 12
            )))
            for _ in range(steps):
                mid = 0.5 * (low + high)
                cand_top, cand_bottom, cand_bad, cand_ratio = trial(mid)
                if cand_bad == 0 and cand_ratio >= min_required:
                    best = (cand_top, cand_bottom, mid, cand_ratio)
                    low = mid
                else:
                    high = mid
            result_top, result_bottom, alpha, min_ratio = best

        # Report the actual final retained-slant continuity after the accepted
        # collective move.  This is also useful for the TinyECL QC text.
        realized = []
        for item in prepared:
            key = item["key"]
            z_top = item["z_top"]
            z_bottom = item["z_bottom"]
            actual_top = self._point_on_coord_pillar(
                result_top[key], result_bottom[key], z_top
            )[:2]
            actual_bottom = self._point_on_coord_pillar(
                result_top[key], result_bottom[key], z_bottom
            )[:2]
            ideal = item["ideal_vector"]
            denom = float(np.dot(ideal, ideal))
            if denom <= 1.0e-18:
                continue
            realized.append(float(np.dot(actual_bottom - actual_top, ideal) / denom))

        max_jump = (
            max(abs(b - a) for a, b in zip(realized[:-1], realized[1:]))
            if len(realized) > 1 else 0.0
        )
        self._fault_plane_straightening_summary = {
            "pillars": len(prepared),
            "accepted_fraction": float(alpha),
            "max_retention_jump": float(max_jump),
            "min_jacobian_ratio": float(min_ratio),
        }
        print(
            "PyGRID: fault-plane straightening: "
            f"regularised {len(prepared)} fault-edge generators; "
            f"collective move fraction = {alpha:.3f}; "
            f"maximum adjacent retained-slant jump = {max_jump:.3f}; "
            f"minimum centre-Jacobian ratio = {min_ratio:.4f}."
        )
        return result_top, result_bottom

    def write_coord(self, f):
        coord_top, coord_bottom = self._coord_pillars()

        # Internal PyGRID geometry intentionally keeps XY in surface/map units
        # and Z in reservoir/depth units.  GRDECL COORD cannot be mixed-unit,
        # so convert XY to the local GRIDUNIT frame only at export.  Z is
        # already in GRIDUNIT because GRIDUNIT follows model.depth_units.
        top_xy = self._map_xy_to_grid_xy(coord_top[:, :, :2])
        bottom_xy = self._map_xy_to_grid_xy(coord_bottom[:, :, :2])

        f.write("COORD\n")
        for j in range(self.model.ny + 1):
            for i in range(self.model.nx + 1):
                xt, yt = top_xy[j, i]
                xb, yb = bottom_xy[j, i]
                zt = coord_top[j, i, 2]
                zb = coord_bottom[j, i, 2]
                f.write(
                    f"  {xt:.6f} {yt:.6f} {zt:.6f} "
                    f"{xb:.6f} {yb:.6f} {zb:.6f}\n"
                )
        f.write("/\n\n")

    def _layer_fractions(self):
        nz = self.model.layers
        if nz is None or nz <= 0:
            raise ValueError("model.layers must be a positive integer")

        thicknesses = self.model.layer_thicknesses
        if thicknesses is None:
            return np.linspace(0.0, 1.0, nz + 1)

        values = np.asarray(thicknesses, dtype=float)
        if values.ndim != 1 or len(values) != nz:
            raise ValueError(
                "model.layer_thicknesses must contain exactly " f"{nz} values"
            )
        if np.any(values <= 0.0):
            raise ValueError("All model.layer_thicknesses values must be > 0")

        cumulative = np.concatenate(([0.0], np.cumsum(values)))
        return cumulative / cumulative[-1]

    def _interpolate(
        self, map_object, xy, barrier_origins=None, fault_set=None
    ):
        if fault_set is None:
            fault_set = self._fault_set_for_level("top")
        if self.model.split_faults and fault_set:
            return map_object.interpolate_faulted(
                xy,
                fault_set,
                search_radius=self.model.search_radius,
                barrier_origins=barrier_origins,
            )
        return map_object.interpolate(xy)

    @staticmethod
    def _cell_corner_queries_from_pillars(pillars):
        """Return four slightly-inset XY queries per logical cell."""
        p = np.asarray(pillars, dtype=float)
        ny = p.shape[0] - 1
        nx = p.shape[1] - 1
        q = np.empty((ny, nx, 4, 2), dtype=float)
        n = GRDECLWriter.FAULT_SIDE_NUDGE

        for j in range(ny):
            for i in range(nx):
                corners = np.asarray(
                    [p[j, i], p[j, i + 1], p[j + 1, i], p[j + 1, i + 1]],
                    dtype=float,
                )
                center = corners.mean(axis=0)
                q[j, i] = corners + n * (center - corners)

        return q

    @staticmethod
    def _cell_centres_from_pillars(pillars):
        p = np.asarray(pillars, dtype=float)
        return 0.25 * (
            p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:]
        )

    def _cell_corner_queries(self):
        """Return TOP XY queries on the original regular pillar grid."""
        return self._cell_corner_queries_from_pillars(self._pillar_array())

    def _bottom_pillar_xy(self):
        """Construct BOTTOM XY by moving FLB fault sticks, then only sideways.

        The TOP grid is never relaxed.  Direct staircase pillars belonging to
        a matched slanted fault first receive their full FLT -> FLB movement.
        Only after those lower fault points are fixed is the *BOTTOM* grid
        equalised on the two sides of each discrete fault face.

        Equalisation is deliberately cross-fault, not radial:

        * X+ fault face -> taper only in +/- I (left/right of that face);
        * Y+ fault face -> taper only in +/- J (left/right of that face).

        There is no relaxation along the strike of the fault.  This prevents a
        local FLB movement from dragging remote pillars along the fault and is
        specifically intended to remove the long triangular spikes produced
        by the earlier circular/radial relaxation field.
        """
        cached = getattr(self, "_bottom_pillar_xy_cache", None)
        if cached is not None:
            return cached

        base = self._pillar_array()
        result = base.copy()

        if (
            not self.model.split_faults
            or not self.model.fault_set
            or not self.model.fault_set.slanted_names
        ):
            self._bottom_pillar_xy_cache = result
            return result

        spacing = self._grid_spacing()
        slanted = set(self.model.fault_set.slanted_names)

        # Build one record per (fault face endpoint, fault name).  The same
        # pillar can appear in several neighbouring staircase faces; records
        # are kept separately until the local spike equaliser has had a chance
        # to compare same-fault stations.
        records_by_key = {}
        records_by_name = {}
        face_records = []

        for name, i1, j1, face in self.fault_faces():
            if name not in slanted:
                continue

            if face == "X+":
                endpoints = ((j1 - 1, i1), (j1, i1))
                normal_axis = "I"
            elif face == "Y+":
                endpoints = ((j1, i1 - 1), (j1, i1))
                normal_axis = "J"
            else:
                continue

            for key in endpoints:
                j, i = key
                base_xy = np.asarray(base[j, i], dtype=float)
                displacement = np.asarray(
                    self.model.fault_set.slant_displacement(
                        name,
                        base_xy,
                        max_top_distance=1.5 * spacing,
                        fade_distance=2.0 * spacing,
                    ),
                    dtype=float,
                )
                if float(np.linalg.norm(displacement)) <= 1.0e-9:
                    continue

                projection = self._slanted_trace_projection(name, base_xy)
                depths = self._fault_reference_depths(
                    name, base_xy, base_xy + displacement
                )

                record = {
                    "key": key,
                    "name": name,
                    "base_xy": base_xy.copy(),
                    "displacement": displacement.copy(),
                    "top_trace": None,
                    "station": None,
                    "depths": depths,
                    "z_fault_top": None,
                    "z_fault_bottom": None,
                    "normal_axis": normal_axis,
                }

                if projection is not None:
                    top_trace, hit = projection
                    record["top_trace"] = top_trace
                    record["station"] = float(hit[2])

                if depths is not None:
                    record["z_fault_top"] = float(depths[0])
                    record["z_fault_bottom"] = float(depths[1])

                records_by_key.setdefault(key, []).append(record)
                records_by_name.setdefault(name, []).append(record)
                face_records.append(record)

        if not records_by_key:
            self._bottom_pillar_xy_cache = result
            return result

        # Re-enable the conservative staircase-corner spike correction before
        # any lower-grid relaxation is performed.  It modifies only a clear
        # duplicate-station outlier and leaves normal fault dip/throw intact.
        self._equalize_slanted_fault_spikes(records_by_name)

        # Exact moved FLB anchors.  At a staircase corner several face records
        # may touch one pillar; use their mean so there is one physical lower
        # endpoint for that COORD stick.
        anchors = {}
        for key, records in records_by_key.items():
            vectors = [
                np.asarray(record["displacement"], dtype=float)
                for record in records
                if float(np.linalg.norm(record["displacement"])) > 1.0e-9
            ]
            if vectors:
                anchors[key] = np.mean(vectors, axis=0)

        if not anchors:
            self._bottom_pillar_xy_cache = result
            return result

        for (j, i), displacement in anchors.items():
            result[j, i] = base[j, i] + displacement

        # In the conforming-grid method, propagate FLT->FLB movement along
        # transverse rows based on the continuous fault station.  This replaces
        # the old alternating I/J relaxation and is what removes the accordion
        # fold at every stair-step turn.
        if bool(getattr(self.model, "conform_slanted_faults", False)):
            bottom_anchor_sets = {}
            for key, records in records_by_key.items():
                candidates = [
                    record for record in records
                    if record.get("top_trace") is not None
                    and record.get("station") is not None
                    and key in anchors
                ]
                if not candidates:
                    continue
                record = candidates[0]
                name = record["name"]
                station = float(np.mean([
                    float(r["station"]) for r in candidates
                    if r.get("station") is not None
                ]))
                trace = record["top_trace"]
                _point, tangent = _point_at_trace_station(trace, station)
                bottom_anchor_sets.setdefault(name, {})[key] = {
                    "target": np.asarray(base[key], dtype=float)
                    + np.asarray(anchors[key], dtype=float),
                    "station": station,
                    "trace": trace,
                    "tangent": np.asarray(tangent, dtype=float),
                }

            if bottom_anchor_sets:
                lines = max(
                    1, int(getattr(self.model, "slanted_fault_conform_lines", 4))
                )
                result = self._propagate_fault_anchor_displacements(
                    base, bottom_anchor_sets, lines
                )
                result = self._guard_conformed_bottom_topology(base, result)
                self._bottom_pillar_xy_cache = result
                moved = int(np.count_nonzero(
                    np.linalg.norm(result - base, axis=2) > 1.0e-9
                ))
                print(
                    "PyGRID: fault-conforming BOTTOM grid: "
                    f"{len(anchors)} FLB anchors; {moved} pillars reshaped "
                    f"through the grid interior."
                )
                return result

        # With ZigZag/equal-width disabled, stop after moving only the direct
        # bottom-fault sticks.  TOP remains untouched in either case.
        if not bool(getattr(self.model, "relax_slanted_faults", True)):
            self._bottom_pillar_xy_cache = result
            return result

        min_lines = max(
            0, int(getattr(self.model, "slanted_fault_relax_lines", 4))
        )
        max_step_fraction = float(
            getattr(self.model, "slanted_fault_max_step_fraction", 0.35)
        )
        if max_step_fraction <= 0.0:
            max_step_fraction = 0.35

        if min_lines <= 0:
            self._bottom_pillar_xy_cache = result
            return result

        ny, nx = self.model.ny, self.model.nx
        contributions = {}
        widest = min_lines

        # Add lower-grid displacement only perpendicular to the discrete fault
        # face.  This is the key difference from the previous radial scheme.
        for record in face_records:
            displacement = np.asarray(record["displacement"], dtype=float)
            magnitude = float(np.linalg.norm(displacement))
            if magnitude <= 1.0e-9:
                continue

            auto_lines = int(
                np.ceil(
                    magnitude
                    / max(max_step_fraction * spacing, 1.0e-12)
                )
            )
            lines = max(min_lines, auto_lines)
            widest = max(widest, lines)
            radius = float(lines + 1)

            j0, i0 = record["key"]
            axis = record["normal_axis"]

            for distance in range(1, lines + 1):
                weight = max(0.0, 1.0 - float(distance) / radius)
                if weight <= 0.0:
                    continue

                for sign in (-1, 1):
                    if axis == "I":
                        key = (j0, i0 + sign * distance)
                    else:
                        key = (j0 + sign * distance, i0)

                    j, i = key
                    if j < 0 or j > ny or i < 0 or i > nx:
                        continue
                    if key in anchors:
                        # Exact FLB fault sticks are authoritative.
                        continue

                    contributions.setdefault(key, []).append(
                        (weight, displacement)
                    )

        relaxed_count = 0
        for key, values in contributions.items():
            weights = np.asarray([item[0] for item in values], dtype=float)
            vectors = np.asarray([item[1] for item in values], dtype=float)
            if not np.any(weights > 0.0):
                continue

            # Average overlapping along-fault/stair-corner contributions, then
            # retain the strongest cross-fault taper at this pillar.
            mean_vector = (
                np.sum(vectors * weights[:, None], axis=0)
                / np.sum(weights)
            )
            taper = float(np.max(weights))
            displacement = mean_vector * taper
            if float(np.linalg.norm(displacement)) <= 1.0e-9:
                continue

            j, i = key
            result[j, i] = base[j, i] + displacement
            relaxed_count += 1

        self._bottom_pillar_xy_cache = result
        print(
            "PyGRID: bottom-only slanted-fault equalisation: moved "
            f"{len(anchors)} FLB fault pillars and relaxed {relaxed_count} "
            f"pillars only left/right across the fault (up to {widest} lines)."
        )
        return result

    def _fault_face_sets(self, fault_set=None, centres=None, cache_tag=None):
        """Return structural-barrier X+/Y+ faces, zero based.

        TOP and BOTTOM may use different fault traces and different XY cell
        centres.  This allows the two endpoint layers to be constructed
        independently before intermediate layers are filled.
        """
        if fault_set is None:
            fault_set = self._fault_set_for_level("top")

        if centres is None:
            centres = self._cell_centres()
            if cache_tag is None:
                cache_tag = "TOP_BASE"
        centres = np.asarray(centres, dtype=float)

        cache_key = None
        if cache_tag is not None:
            cache_key = (str(cache_tag), self._fault_set_cache_key(fault_set))
            cached = self._surface_fault_face_sets_cache.get(cache_key)
            if cached is not None:
                return cached

        x_faces = set()
        y_faces = set()

        if not self.model.split_faults or not fault_set:
            result = (x_faces, y_faces)
            if cache_key is not None:
                self._surface_fault_face_sets_cache[cache_key] = result
            return result

        # A conforming COORD/MIXED grid keeps one logical fault path through all K
        # layers.  Use the raw row/column crossings selected before deformation
        # rather than re-intersecting a moved cell-centre grid.
        if (
            bool(getattr(self.model, "conform_slanted_faults", False))
            and self._slanted_fault_mode() in {"COORD", "MIXED"}
        ):
            allowed_names = {trace.name for trace in fault_set.traces}
            for name, i1, j1, face in self._raw_fault_faces():
                if name not in allowed_names:
                    continue
                if face == "X+":
                    x_faces.add((i1 - 1, j1 - 1))
                elif face == "Y+":
                    y_faces.add((i1 - 1, j1 - 1))
            result = (x_faces, y_faces)
            if cache_key is not None:
                self._surface_fault_face_sets_cache[cache_key] = result
            return result

        nx, ny = self.model.nx, self.model.ny

        for trace in fault_set.traces:
            if not getattr(trace, "surface_barrier", True):
                continue

            for f1, f2 in trace.segments:
                for j in range(ny):
                    for i in range(nx - 1):
                        if segment_intersection(
                            centres[j, i], centres[j, i + 1], f1, f2
                        ) is not None:
                            x_faces.add((i, j))

                for j in range(ny - 1):
                    for i in range(nx):
                        if segment_intersection(
                            centres[j, i], centres[j + 1, i], f1, f2
                        ) is not None:
                            y_faces.add((i, j))

        result = (x_faces, y_faces)
        if cache_key is not None:
            self._surface_fault_face_sets_cache[cache_key] = result
        return result

    def _split_corner_topology(
        self, fault_set=None, centres=None, cache_tag=None
    ):
        """
        Build structural smoothing nodes from surface-barrier fault faces.

        Every logical cell owns four ZCORN corner values.  Corners at the same
        physical pillar are merged across ordinary faces and across fault pieces
        classified as zero throw.  They remain separate only across faults that
        are structural-surface barriers.
        """
        ny, nx = self.model.ny, self.model.nx
        n_nodes = ny * nx * 4
        parent = np.arange(n_nodes, dtype=int)

        def node(j, i, corner):
            return ((j * nx + i) * 4) + corner

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        x_faults, y_faults = self._fault_face_sets(
            fault_set, centres=centres, cache_tag=cache_tag
        )

        # Across I: left NE/SE == right NW/SW unless the X+ face is faulted.
        for j in range(ny):
            for i in range(nx - 1):
                if (i, j) in x_faults:
                    continue
                union(node(j, i, 1), node(j, i + 1, 0))
                union(node(j, i, 3), node(j, i + 1, 2))

        # Across J: upper SW/SE == lower NW/NE unless the Y+ face is faulted.
        for j in range(ny - 1):
            for i in range(nx):
                if (i, j) in y_faults:
                    continue
                union(node(j, i, 2), node(j + 1, i, 0))
                union(node(j, i, 3), node(j + 1, i, 1))

        roots = np.asarray([find(i) for i in range(n_nodes)], dtype=int)
        unique, inverse = np.unique(roots, return_inverse=True)
        groups = inverse.reshape((ny, nx, 4))

        # Build a local-neighbour graph between the resulting split pillar nodes.
        # Only edges *inside* a cell are added.  Because duplicate corners were
        # not merged across a fault face, this graph cannot cross a fault.
        neighbours = [set() for _ in range(len(unique))]
        for j in range(ny):
            for i in range(nx):
                g = groups[j, i]
                for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
                    ga, gb = int(g[a]), int(g[b])
                    if ga != gb:
                        neighbours[ga].add(gb)
                        neighbours[gb].add(ga)

        return groups, neighbours

    @staticmethod
    def _group_average(values, groups):
        """Average duplicate cell-corner values that belong to one split node."""
        flat_values = np.asarray(values, dtype=float).reshape(-1)
        flat_groups = np.asarray(groups, dtype=int).reshape(-1)
        count = int(flat_groups.max()) + 1
        sums = np.bincount(flat_groups, weights=flat_values, minlength=count)
        nums = np.bincount(flat_groups, minlength=count)
        return sums / np.maximum(nums, 1)

    @staticmethod
    def _laplacian_pass(values, neighbours, factor):
        """Apply one connectivity-aware Laplacian displacement pass."""
        source = np.asarray(values, dtype=float)
        updated = source.copy()
        for idx, adjacent in enumerate(neighbours):
            if not adjacent:
                continue
            mean = float(np.mean(source[list(adjacent)]))
            updated[idx] = source[idx] + factor * (mean - source[idx])
        return updated

    @staticmethod
    def _bilateral_pass(values, neighbours, factor, sigma_z):
        """Apply one fault-connectivity-aware bilateral smoothing pass.

        Only directly connected split-grid neighbours participate.  Their
        influence is reduced exponentially when their Z differs strongly from
        the current node.  This suppresses small local waviness without
        aggressively flattening genuine structural relief.
        """
        source = np.asarray(values, dtype=float)
        updated = source.copy()
        sigma_z = float(sigma_z)
        if sigma_z <= 0.0:
            raise ValueError("model.smooth_bilateral_sigma_z must be > 0")

        inv_two_sigma2 = 1.0 / (2.0 * sigma_z * sigma_z)
        for idx, adjacent in enumerate(neighbours):
            if not adjacent:
                continue
            ids = np.fromiter(adjacent, dtype=int)
            dz = source[ids] - source[idx]
            weights = np.exp(-(dz * dz) * inv_two_sigma2)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1.0e-15:
                continue
            mean = float(np.sum(weights * source[ids]) / weight_sum)
            updated[idx] = source[idx] + factor * (mean - source[idx])
        return updated


    @staticmethod
    def _biharmonic_smooth(values, neighbours, strength):
        """Curvature-minimising smoothing on the split-node graph.

        Solve (I + strength * L.T * L) z = z0, where L is the graph
        Laplacian of the already fault-split ZCORN topology.  Because there are
        no graph edges across fault faces, the solve cannot smooth through a
        fault.  The L.T*L term penalises curvature rather than slope, so broad
        structural trends are preserved better than with repeated neighbour
        averaging while terrace-like contour imprint is relaxed.
        """
        source = np.asarray(values, dtype=float)
        strength = float(strength)
        if strength <= 0.0:
            return source.copy()

        n = len(source)
        rows = []
        cols = []
        data = []
        for idx, adjacent in enumerate(neighbours):
            degree = len(adjacent)
            rows.append(idx); cols.append(idx); data.append(float(degree))
            for other in adjacent:
                rows.append(idx); cols.append(int(other)); data.append(-1.0)

        L = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        A = sparse.eye(n, format='csr') + strength * (L.T @ L)
        return np.asarray(spsolve(A, source), dtype=float)

    @staticmethod
    def _automatic_bilateral_sigma(values, neighbours):
        """Choose a robust Z scale from connected-neighbour differences."""
        source = np.asarray(values, dtype=float)
        diffs = []
        for idx, adjacent in enumerate(neighbours):
            for other in adjacent:
                if other > idx:
                    diffs.append(abs(float(source[other] - source[idx])))
        if not diffs:
            return 1.0
        # The upper quartile follows the typical local structural gradient but
        # avoids letting the largest genuine relief control the filter.
        sigma = 1.5 * float(np.percentile(np.asarray(diffs), 75.0))
        return max(sigma, 1.0e-6)

    def _smooth_surface(
        self, surface, fault_set=None, centres=None, cache_tag=None
    ):
        """Smooth one already fault-split structural surface.

        The split-node topology is created from the selected endpoint fault faces.
        Duplicate cell corners are merged only where cells are genuinely
        connected.  Therefore every smoothing method operates independently
        on the two sides of a fault and cannot reduce the fault throw by
        averaging through the fault face.
        """
        iterations = int(getattr(self.model, "smooth_iterations", 0) or 0)
        factor = float(getattr(self.model, "smooth_factor", 0.15) or 0.0)

        groups, neighbours = self._split_corner_topology(
            fault_set, centres=centres, cache_tag=cache_tag
        )
        node_values = self._group_average(surface, groups)

        if iterations <= 0 or factor <= 0.0:
            return node_values[groups]
        if factor > 1.0:
            raise ValueError("model.smooth_factor must be between 0 and 1")

        method = str(
            getattr(self.model, "smoothing_method", "LAPLACIAN") or "LAPLACIAN"
        ).upper()
        if method not in {"LAPLACIAN", "TAUBIN", "BILATERAL", "BIHARMONIC"}:
            raise ValueError(
                "model.smoothing_method must be 'LAPLACIAN', 'TAUBIN', "
                "'BILATERAL', or 'BIHARMONIC'"
            )

        if method == "LAPLACIAN":
            for _ in range(iterations):
                node_values = self._laplacian_pass(
                    node_values, neighbours, factor
                )

        elif method == "TAUBIN":
            mu = getattr(self.model, "smooth_taubin_mu", None)
            if mu is None:
                mu = -1.05 * factor
            mu = float(mu)
            if mu >= 0.0:
                raise ValueError("model.smooth_taubin_mu must be negative")
            if abs(mu) > 1.0:
                raise ValueError("abs(model.smooth_taubin_mu) must be <= 1")

            for _ in range(iterations):
                node_values = self._laplacian_pass(
                    node_values, neighbours, factor
                )
                node_values = self._laplacian_pass(
                    node_values, neighbours, mu
                )

        elif method == "BILATERAL":
            sigma_z = getattr(self.model, "smooth_bilateral_sigma_z", None)
            if sigma_z is None:
                sigma_z = self._automatic_bilateral_sigma(
                    node_values, neighbours
                )
            sigma_z = float(sigma_z)
            for _ in range(iterations):
                node_values = self._bilateral_pass(
                    node_values, neighbours, factor, sigma_z
                )

        else:  # BIHARMONIC
            strength = getattr(self.model, "smooth_biharmonic_lambda", None)
            if strength is None:
                strength = max(1.0e-6, iterations * factor * 2.0)
            node_values = self._biharmonic_smooth(
                node_values, neighbours, strength
            )

        return node_values[groups]

    def _connected_corner_average(
        self, values, fault_set=None, centres=None, cache_tag=None
    ):
        """Make duplicate corners identical without spatial smoothing.

        This is useful for thickness maps: a thickness value may vary in space,
        but corners belonging to one ordinary (non-faulted) pillar must agree.
        Across a fault the split corners remain independent.
        """
        groups, _ = self._split_corner_topology(
            fault_set, centres=centres, cache_tag=cache_tag
        )
        nodes = self._group_average(values, groups)
        return nodes[groups]

    def _query_map_at_cell_corners(
        self, map_object, flat, barrier_origins, fault_set=None
    ):
        values = np.asarray(
            self._interpolate(
                map_object,
                flat,
                barrier_origins=barrier_origins,
                fault_set=fault_set,
            ),
            dtype=float,
        )
        return values.reshape((self.model.ny, self.model.nx, 4))

    def _cache_interfaces(self, interfaces):
        self._interfaces_cache = np.asarray(interfaces, dtype=float)
        return self._interfaces_cache

    @staticmethod
    def _pillar_adjacent_corner_indices(j, i, ny, nx):
        """Yield (cell_j, cell_i, corner) entries touching one pillar."""
        if j < ny and i < nx:
            yield j, i, 0          # NW
        if j < ny and i > 0:
            yield j, i - 1, 1      # NE
        if j > 0 and i < nx:
            yield j - 1, i, 2      # SW
        if j > 0 and i > 0:
            yield j - 1, i - 1, 3  # SE

    def _pillar_interface_depths(self, key, interfaces=None):
        """Return robust TOP/BOTTOM endpoint depths at one logical pillar."""
        if interfaces is None:
            interfaces = self._cell_corner_interfaces()

        j, i = key
        ny, nx = self.model.ny, self.model.nx
        top_values = []
        bottom_values = []
        for cj, ci, corner in self._pillar_adjacent_corner_indices(
            j, i, ny, nx
        ):
            top_values.append(float(interfaces[0, cj, ci, corner]))
            bottom_values.append(float(interfaces[-1, cj, ci, corner]))

        top_values = np.asarray(top_values, dtype=float)
        bottom_values = np.asarray(bottom_values, dtype=float)
        top_values = top_values[np.isfinite(top_values)]
        bottom_values = bottom_values[np.isfinite(bottom_values)]
        if len(top_values) == 0 or len(bottom_values) == 0:
            return None

        z_top = float(np.median(top_values))
        z_bottom = float(np.median(bottom_values))
        if abs(z_bottom - z_top) <= 1.0e-6:
            return None
        return z_top, z_bottom

    def _cell_corner_interfaces(self):
        """Build TOP, then BOTTOM, then fill all interfaces between them.

        Fault handling follows TinyECL CreateGrid exactly:

        * Split Grid OFF: neither endpoint uses fault barriers.
        * ZigZag ON: TOP and BOTTOM use the same TOP (*.flt) fault traces, so
          faults remain vertical through K and *.flb is deliberately ignored.
        * ZigZag OFF: TOP uses *.flt and BOTTOM uses the matched *.flb
          interpretation on the displaced lower pillar geometry.

        Only after both endpoint horizons are fixed are the K interfaces filled
        by the requested proportional/absolute layer definition.
        """
        if self._interfaces_cache is not None:
            return self._interfaces_cache
        if not hasattr(self.model, "top_map"):
            raise ValueError("model.top must be set before writing ZCORN")

        ny, nx, nz = self.model.ny, self.model.nx, self.model.layers
        if nz is None or nz <= 0:
            raise ValueError("model.layers must be a positive integer")

        top_pillars = self._pillar_array()
        top_queries = self._cell_corner_queries_from_pillars(top_pillars)
        top_flat = top_queries.reshape((-1, 2))
        top_centres = self._cell_centres_from_pillars(top_pillars)
        top_origins = np.repeat(
            top_centres[:, :, None, :], 4, axis=2
        ).reshape((-1, 2))
        top_faults = self._fault_set_for_level("top")

        # 1) Structural TOP: FLT + original grid.
        top = self._query_map_at_cell_corners(
            self.model.top_map,
            top_flat,
            top_origins,
            fault_set=top_faults,
        )
        top = self._smooth_surface(
            top,
            fault_set=top_faults,
            centres=top_centres,
            cache_tag="TOP",
        )

        mode = str(getattr(self.model, "layer_mode", "") or "").upper()
        fractions = self._layer_fractions()

        # ------------------------------------------------------------
        # 2) Structural BOTTOM: build the lower XY grid from FLB first,
        #    then interpolate/smooth BOTTOM on that exact geometry.
        # 3) Fill all internal interfaces only after both endpoints exist.
        # ------------------------------------------------------------
        if mode in {"", "PROPORTIONAL", "BOTTOM"} and hasattr(
            self.model, "bottom_map"
        ):
            if self._slanted_fault_mode() == "LAYERED":
                # Keep the same regular XY pillar grid at BOTTOM.  Slanted
                # geometry is represented by K-dependent fault-face migration,
                # not by tilting COORD sticks.
                bottom_pillars = top_pillars
            else:
                bottom_pillars = self._bottom_pillar_xy()

            bottom_queries = self._cell_corner_queries_from_pillars(
                bottom_pillars
            )
            bottom_flat = bottom_queries.reshape((-1, 2))
            bottom_centres = self._cell_centres_from_pillars(bottom_pillars)
            bottom_origins = np.repeat(
                bottom_centres[:, :, None, :], 4, axis=2
            ).reshape((-1, 2))
            bottom_faults = self._fault_set_for_level("bottom")

            bottom = self._query_map_at_cell_corners(
                self.model.bottom_map,
                bottom_flat,
                bottom_origins,
                fault_set=bottom_faults,
            )
            bottom = self._smooth_surface(
                bottom,
                fault_set=bottom_faults,
                centres=bottom_centres,
                cache_tag="BOTTOM",
            )

            minimum_zone_thickness = float(
                getattr(self.model, "minimum_zone_thickness", 0.0) or 0.0
            )
            if minimum_zone_thickness < 0.0:
                raise ValueError("model.minimum_zone_thickness must be >= 0")

            if not self.model.split_faults:
                fault_message = "faults disabled by Split Grid"
            elif self._slanted_fault_mode() == "LAYERED":
                fault_message = "ZigZag: all TOP faults vertical"
            elif self._slanted_fault_mode() == "MIXED":
                names = ",".join(getattr(self.model.fault_set, "slanted_names", []))
                fault_message = (
                    "ZigZag mixed: TOP-only faults zigzag; "
                    f"FLT/FLB faults conform/slant ({names or 'none'})"
                )
            else:
                if getattr(self.model.fault_set, "slanted_names", []):
                    fault_message = (
                        "conforming: all TOP faults from FLT; matching FLB faults slanted"
                    )
                else:
                    fault_message = "conforming: all TOP faults from FLT; no FLB slant"

            print(
                "PyGRID: TOP/BOTTOM-first ZCORN: "
                f"{fault_message}; filling internal K layers."
            )

            separation = bottom - top
            pinched = separation < minimum_zone_thickness
            if np.any(pinched):
                separation = np.maximum(separation, minimum_zone_thickness)
                bottom = top + separation
                print(
                    "PyGRID: structural zone pinched/truncated at "
                    f"{int(np.count_nonzero(pinched))} cell corners "
                    f"(minimum zone thickness = {minimum_zone_thickness:g})."
                )

            split_layer = int(
                getattr(self.model, "thickness_split_layer", 0) or 0
            )
            thickness_mode = str(
                getattr(self.model, "thickness_mode", "") or ""
            ).upper()
            use_mid = (
                split_layer > 0
                and split_layer < nz
                and hasattr(self.model, "thickness_map")
            )

            if use_mid:
                if thickness_mode in {"", "FROM_TOP"}:
                    thickness = self._query_map_at_cell_corners(
                        self.model.thickness_map,
                        top_flat,
                        top_origins,
                        fault_set=top_faults,
                    )
                    thickness = self._connected_corner_average(
                        thickness,
                        fault_set=top_faults,
                        centres=top_centres,
                        cache_tag="TOP",
                    )
                    if np.any(thickness <= 0.0):
                        raise ValueError(
                            "Interpolated thickness must be positive everywhere; "
                            f"minimum is {float(np.min(thickness))}"
                        )
                    mid = self._smooth_surface(
                        top + thickness,
                        fault_set=top_faults,
                        centres=top_centres,
                        cache_tag="TOP",
                    )
                else:
                    raise ValueError(
                        "model.thickness_mode currently supports only "
                        "'FROM_TOP' for an internal thickness horizon"
                    )

                lower_limit = top + minimum_zone_thickness
                upper_limit = bottom - minimum_zone_thickness
                invalid_zone = upper_limit < lower_limit
                if np.any(invalid_zone):
                    midpoint = 0.5 * (top + bottom)
                    lower_limit = np.where(invalid_zone, midpoint, lower_limit)
                    upper_limit = np.where(invalid_zone, midpoint, upper_limit)

                mid_clipped = np.clip(mid, lower_limit, upper_limit)
                clipped = np.abs(mid_clipped - mid) > 1.0e-9
                if np.any(clipped):
                    print(
                        "PyGRID: thickness horizon clipped to TOP/BOTTOM at "
                        f"{int(np.count_nonzero(clipped))} cell corners."
                    )
                mid = mid_clipped

                weights = np.asarray(self.model.layer_thicknesses, dtype=float)
                if weights.ndim != 1 or len(weights) != nz:
                    raise ValueError(
                        "model.layer_thicknesses must contain exactly "
                        f"{nz} values"
                    )
                if np.any(weights <= 0.0):
                    raise ValueError(
                        "All model.layer_thicknesses values must be > 0"
                    )

                upper_weights = weights[:split_layer]
                lower_weights = weights[split_layer:]
                upper_cum = np.concatenate(([0.0], np.cumsum(upper_weights)))
                lower_cum = np.concatenate(([0.0], np.cumsum(lower_weights)))
                upper_frac = upper_cum / upper_cum[-1]
                lower_frac = lower_cum / lower_cum[-1]

                interfaces = np.empty((nz + 1, ny, nx, 4), dtype=float)
                upper_sep = mid - top
                for k in range(split_layer + 1):
                    interfaces[k] = top + upper_sep * upper_frac[k]

                lower_sep = bottom - mid
                for local_k in range(1, len(lower_frac)):
                    k = split_layer + local_k
                    interfaces[k] = mid + lower_sep * lower_frac[local_k]

                return self._cache_interfaces(interfaces)

            interfaces = np.empty((nz + 1, ny, nx, 4), dtype=float)
            for k, fraction in enumerate(fractions):
                interfaces[k] = top + separation * fraction
            return self._cache_interfaces(interfaces)

        # TOP + THICKNESS map.  No independent BOTTOM interpretation exists,
        # so the TOP fault topology remains authoritative throughout.
        if mode == "THICKNESS" or (
            mode == "PROPORTIONAL"
            and not hasattr(self.model, "bottom_map")
            and hasattr(self.model, "thickness_map")
        ):
            if not hasattr(self.model, "thickness_map"):
                raise ValueError(
                    "model.layer_mode='THICKNESS' requires model.thickness"
                )
            thickness = self._query_map_at_cell_corners(
                self.model.thickness_map,
                top_flat,
                top_origins,
                fault_set=top_faults,
            )
            thickness = self._connected_corner_average(
                thickness,
                fault_set=top_faults,
                centres=top_centres,
                cache_tag="TOP",
            )
            if np.any(thickness <= 0.0):
                raise ValueError(
                    "Interpolated thickness must be positive everywhere; "
                    f"minimum is {float(np.min(thickness))}"
                )

            interfaces = np.empty((nz + 1, ny, nx, 4), dtype=float)
            for k, fraction in enumerate(fractions):
                interfaces[k] = top + thickness * fraction
            return self._cache_interfaces(interfaces)

        # Layer-cake fallback.  With no bottom/thickness structure,
        # layer_thicknesses are absolute DZ values in model.depth_units.
        if mode in {"ABSOLUTE", "DZ"}:
            thicknesses = self.model.layer_thicknesses
            if thicknesses is None:
                raise ValueError(
                    "Absolute layer-cake mode requires model.layer_thicknesses"
                )
            values = np.asarray(thicknesses, dtype=float)
            if values.ndim != 1 or len(values) != nz:
                raise ValueError(
                    "model.layer_thicknesses must contain exactly "
                    f"{nz} values"
                )
            if np.any(values <= 0.0):
                raise ValueError(
                    "All absolute layer thickness values must be > 0"
                )

            interfaces = np.empty((nz + 1, ny, nx, 4), dtype=float)
            interfaces[0] = top
            cumulative = 0.0
            for k, dz in enumerate(values, start=1):
                cumulative += float(dz)
                interfaces[k] = top + cumulative
            return self._cache_interfaces(interfaces)

        # Backwards-compatible top+thickness fallback.
        if hasattr(self.model, "thickness_map"):
            thickness = self._query_map_at_cell_corners(
                self.model.thickness_map,
                top_flat,
                top_origins,
                fault_set=top_faults,
            )
            thickness = self._connected_corner_average(
                thickness,
                fault_set=top_faults,
                centres=top_centres,
                cache_tag="TOP",
            )
            if np.any(thickness <= 0.0):
                raise ValueError(
                    "Interpolated thickness must be positive everywhere; "
                    f"minimum is {float(np.min(thickness))}"
                )
            interfaces = np.empty((nz + 1, ny, nx, 4), dtype=float)
            for k, fraction in enumerate(fractions):
                interfaces[k] = top + thickness * fraction
            return self._cache_interfaces(interfaces)

        raise ValueError(
            "Unable to build ZCORN: define a bottom structure, a thickness "
            "map, or use ABSOLUTE layer-cake mode."
        )

    @staticmethod
    def _zcorn_from_cell_interfaces(interfaces):
        """Convert independent cell-corner interfaces to Eclipse ZCORN order."""
        nz = interfaces.shape[0] - 1
        ny = interfaces.shape[1]
        nx = interfaces.shape[2]
        zcorn = np.empty((2 * nz, 2 * ny, 2 * nx), dtype=float)

        # corners: 0=NW, 1=NE, 2=SW, 3=SE
        for k in range(nz):
            top = interfaces[k]
            bot = interfaces[k + 1]

            zcorn[2 * k, 0::2, 0::2] = top[:, :, 0]
            zcorn[2 * k, 0::2, 1::2] = top[:, :, 1]
            zcorn[2 * k, 1::2, 0::2] = top[:, :, 2]
            zcorn[2 * k, 1::2, 1::2] = top[:, :, 3]

            zcorn[2 * k + 1, 0::2, 0::2] = bot[:, :, 0]
            zcorn[2 * k + 1, 0::2, 1::2] = bot[:, :, 1]
            zcorn[2 * k + 1, 1::2, 0::2] = bot[:, :, 2]
            zcorn[2 * k + 1, 1::2, 1::2] = bot[:, :, 3]

        return zcorn.ravel(order="C")

    def write_zcorn(self, f):
        interfaces = self._cell_corner_interfaces()
        values = self._zcorn_from_cell_interfaces(interfaces)

        f.write("ZCORN\n")
        for start in range(0, len(values), 8):
            chunk = values[start:start + 8]
            f.write("  " + " ".join(f"{value:.6f}" for value in chunk) + "\n")
        f.write("/\n")

    def _cell_centres(self):
        """Return TOP XY centres for all logical grid cells."""
        return self._cell_centres_from_pillars(self._pillar_array())

    def _faces_for_fault_set(self, fault_set):
        """Convert one XY fault trace set to stair-step Eclipse cell faces."""
        return self._faces_for_fault_set_on_centres(
            fault_set, self._cell_centres()
        )

    def faces_for_fault_set(self, fault_set):
        """Public wrapper used for FAULTS-only transmissibility output."""
        return self._faces_for_fault_set(fault_set)

    def layered_fault_faces(self):
        """Return K-dependent FAULTS records for LAYERED slanted faults.

        Each record is ``(name, i, j, k1, k2, face)``.

        The first K layer uses the exact FLT stair-step.  The last K layer uses
        the support-aware FLB interpretation: portions covered by FLB migrate
        downward while unsupported FLT tails remain vertical.  Intermediate
        layers migrate continuously between those endpoints.  Adjacent K layers
        using the same I/J face are compacted into one K range.
        """
        nz = int(self.model.layers or 0)
        if nz <= 0:
            return []

        if (
            self._slanted_fault_mode() != "LAYERED"
            or not self.model.split_faults
            or not self.model.fault_set
            or not self.model.fault_set.slanted_names
        ):
            return [
                (name, i, j, 1, nz, face)
                for name, i, j, face in self.fault_faces()
            ]

        if nz == 1:
            fractions = np.asarray([0.0], dtype=float)
        else:
            interfaces = self._layer_fractions()
            mids = 0.5 * (interfaces[:-1] + interfaces[1:])
            span = float(mids[-1] - mids[0])
            if span <= 1.0e-12:
                fractions = np.linspace(0.0, 1.0, nz)
            else:
                fractions = (mids - mids[0]) / span
                fractions[0] = 0.0
                fractions[-1] = 1.0

        face_layers = {}
        for k, fraction in enumerate(fractions, start=1):
            fault_set = self._fault_set_for_fraction(float(fraction))
            for name, i, j, face in self._faces_for_fault_set(fault_set):
                face_layers.setdefault((name, i, j, face), []).append(k)

        records = []
        for (name, i, j, face), layers in face_layers.items():
            layers = sorted(set(layers))
            k1 = k2 = layers[0]
            for k in layers[1:]:
                if k == k2 + 1:
                    k2 = k
                else:
                    records.append((name, i, j, k1, k2, face))
                    k1 = k2 = k
            records.append((name, i, j, k1, k2, face))

        records.sort(key=lambda x: (x[0], x[2], x[1], x[3], x[5]))
        print(
            "PyGRID: layered slanted faults: vertical COORD pillars; "
            f"{len(records)} K-dependent support-aware FAULTS records."
        )
        return records

    def fault_faces(self):
        """
        Convert digitized faults to stair-step Eclipse cell faces.

        A fault face is inserted wherever a digitized fault crosses the line
        between two neighbouring cell centres (the dual grid).  Therefore the
        discrete fault always lies *between* cells.  No cell is split internally
        and no cell may bridge from the low block to the high block.
        """
        if self._fault_faces_cache is not None:
            return self._fault_faces_cache
        if not self.model.split_faults or not self.model.fault_set:
            self._fault_faces_cache = []
            return []

        if bool(getattr(self.model, "conform_slanted_faults", False)) and self._slanted_fault_mode() in {"COORD", "MIXED"}:
            self._fault_faces_cache = list(self._raw_fault_faces())
        else:
            self._fault_faces_cache = self._faces_for_fault_set(
                self._fault_set_for_level("top")
            )
        return self._fault_faces_cache

    @staticmethod
    def _point_on_coord_pillar(top_point, bottom_point, z):
        """Return XYZ at depth ``z`` on one straight COORD pillar."""
        top_point = np.asarray(top_point, dtype=float)
        bottom_point = np.asarray(bottom_point, dtype=float)
        dz = float(bottom_point[2] - top_point[2])
        if abs(dz) <= 1.0e-12:
            return np.asarray([top_point[0], top_point[1], float(z)], dtype=float)
        fraction = (float(z) - float(top_point[2])) / dz
        xy = top_point[:2] + fraction * (bottom_point[:2] - top_point[:2])
        return np.asarray([xy[0], xy[1], float(z)], dtype=float)

    def _slanted_cell_center_jacobians(self):
        """Return centre Jacobian determinants for the complete corner grid.

        This is a compact geometry QC, not a simulator-volume calculation.  A
        sign reversal means the trilinear hexahedron is locally inverted at its
        centre and should never be written silently as a production grid.
        """
        coord_top, coord_bottom = self._coord_pillars()
        return self._cell_center_jacobians_from_coords(coord_top, coord_bottom)

    def _validate_slanted_cell_geometry(self):
        """Reject inverted COORD cells before writing a slanted GRDECL."""
        if not bool(getattr(self.model, "validate_slanted_geometry", True)):
            return
        if (
            self._slanted_fault_mode() not in {"COORD", "MIXED"}
            or not self.model.split_faults
            or not self.model.fault_set
            or not self.model.fault_set.traces
        ):
            return

        dets = self._slanted_cell_center_jacobians()
        pinched = self._pinched_cell_mask()
        active_finite = np.isfinite(dets) & ~pinched
        finite = dets[active_finite]
        pinched_count = int(np.count_nonzero(pinched))
        if len(finite) == 0:
            if pinched_count and np.all(np.isfinite(dets)):
                print(
                    "PyGRID: slanted geometry QC: 0 inverted cells; "
                    f"{pinched_count} intentionally pinched/collapsed cells."
                )
                return
            raise ValueError("Slanted-fault geometry QC produced no finite active cells")

        median = float(np.median(finite))
        orientation = 1.0 if median >= 0.0 else -1.0
        signed = orientation * dets
        bad_mask = (
            ~np.isfinite(signed)
            | ((signed <= 0.0) & ~pinched)
        )
        bad = np.argwhere(bad_mask)
        if len(bad):
            sample = ", ".join(
                f"I={int(i)+1},J={int(j)+1},K={int(k)+1}"
                for k, j, i in bad[:10]
            )
            raise ValueError(
                "Slanted-fault geometry would create "
                f"{len(bad)} inverted grid cells ({sample}). "
                "The GRDECL was not written.  Check the FLT/FLB interpretation "
                "or reduce the local fault dip."
            )

        median_abs = max(abs(median), 1.0e-30)
        active_signed = signed[~pinched & np.isfinite(signed)]
        min_ratio = float(np.min(active_signed)) / median_abs
        message = (
            "PyGRID: slanted geometry QC: 0 inverted cells; "
            f"minimum active-cell centre-Jacobian ratio = {min_ratio:.4f}"
        )
        if pinched_count:
            message += f"; {pinched_count} intentionally pinched/collapsed cells"
        print(message + ".")


    def _fault_face_graph_stats(self, faces):
        """Return connected-component and branch statistics for fault edges."""
        adjacency = {}
        for record in faces:
            if len(record) == 4:
                _name, i1, j1, face = record
            else:
                i1, j1, face = record[-3:]
            endpoints = self._fault_edge_endpoints(int(i1), int(j1), face)
            if len(endpoints) != 2:
                continue
            a, b = endpoints
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        unseen = set(adjacency)
        components = 0
        while unseen:
            components += 1
            stack = [next(iter(unseen))]
            while stack:
                node = stack.pop()
                if node not in unseen:
                    continue
                unseen.remove(node)
                stack.extend(adjacency.get(node, ()))
        branch_nodes = sum(1 for neighbours in adjacency.values() if len(neighbours) > 2)
        end_nodes = sum(1 for neighbours in adjacency.values() if len(neighbours) == 1)
        return {
            "components": int(components),
            "branch_nodes": int(branch_nodes),
            "end_nodes": int(end_nodes),
            "vertices": int(len(adjacency)),
        }

    def _fault_block_count(self, faces):
        """Count connected I/J cell regions after removing selected fault faces."""
        nx, ny = self.model.nx, self.model.ny
        barriers = {(int(i1), int(j1), str(face)) for _n, i1, j1, face in faces}
        seen = set()
        blocks = 0
        for j in range(ny):
            for i in range(nx):
                seed = (j, i)
                if seed in seen:
                    continue
                blocks += 1
                stack = [seed]
                seen.add(seed)
                while stack:
                    cj, ci = stack.pop()
                    neighbours = []
                    if ci + 1 < nx and (ci + 1, cj + 1, "X+") not in barriers:
                        neighbours.append((cj, ci + 1))
                    if ci - 1 >= 0 and (ci, cj + 1, "X+") not in barriers:
                        neighbours.append((cj, ci - 1))
                    if cj + 1 < ny and (ci + 1, cj + 1, "Y+") not in barriers:
                        neighbours.append((cj + 1, ci))
                    if cj - 1 >= 0 and (ci + 1, cj, "Y+") not in barriers:
                        neighbours.append((cj - 1, ci))
                    for node in neighbours:
                        if node not in seen:
                            seen.add(node)
                            stack.append(node)
        return int(blocks)

    def _geometric_fault_faces_from_interfaces(self, interfaces=None):
        """Reconstruct internal ZCORN discontinuity faces from final geometry.

        This mirrors the useful PyGRDECL idea of detecting faults from an
        already-built corner-point grid rather than trusting the input fault
        definition.  Names cannot be reconstructed from geometry alone, so the
        returned records are ``(I, J, FACE)`` logical locations.
        """
        if interfaces is None:
            interfaces = self._cell_corner_interfaces()
        z = np.asarray(interfaces, dtype=float)
        tol = max(
            0.0,
            float(getattr(self.model, "fault_topology_qc_depth_tolerance", 1.0e-6)),
        )
        nx, ny = self.model.nx, self.model.ny
        detected = set()

        for j in range(ny):
            for i in range(nx - 1):
                left = z[:, j, i, :]
                right = z[:, j, i + 1, :]
                diff = np.concatenate((
                    np.abs(left[:, 1] - right[:, 0]),
                    np.abs(left[:, 3] - right[:, 2]),
                ))
                finite = diff[np.isfinite(diff)]
                if finite.size and float(np.max(finite)) > tol:
                    detected.add((i + 1, j + 1, "X+"))

        for j in range(ny - 1):
            for i in range(nx):
                upper = z[:, j, i, :]
                lower = z[:, j + 1, i, :]
                diff = np.concatenate((
                    np.abs(upper[:, 2] - lower[:, 0]),
                    np.abs(upper[:, 3] - lower[:, 1]),
                ))
                finite = diff[np.isfinite(diff)]
                if finite.size and float(np.max(finite)) > tol:
                    detected.add((i + 1, j + 1, "Y+"))

        return sorted(detected, key=lambda x: (x[1], x[0], x[2]))

    def _fault_topology_qc_summary(self):
        """Independent V4 fault topology / fault-block QC summary."""
        if (
            not bool(getattr(self.model, "fault_topology_qc", True))
            or not self.model.split_faults
            or not self.model.fault_set
        ):
            return {"enabled": False}
        intended = list(self._raw_fault_faces())
        intended_locations = {(int(i), int(j), face) for _n, i, j, face in intended}
        geometric = self._geometric_fault_faces_from_interfaces()
        geometric_locations = set(geometric)
        matched = intended_locations & geometric_locations
        missing = intended_locations - geometric_locations
        unexpected = geometric_locations - intended_locations

        barrier_names = set()
        if self.model.fault_set:
            for name in self.model.fault_set.names:
                named = self.model.fault_set.named_traces(name, bottom=False)
                if named and all(bool(getattr(t, "surface_barrier", True)) for t in named):
                    barrier_names.add(name)
        expected_barrier_locations = {
            (int(i), int(j), face)
            for name, i, j, face in intended
            if name in barrier_names
        }
        missing_barrier = expected_barrier_locations - geometric_locations

        intended_stats = self._fault_face_graph_stats(intended)
        geometric_named = [("GEOMETRY", i, j, face) for i, j, face in geometric]
        geometric_stats = self._fault_face_graph_stats(geometric_named)
        summary = {
            "enabled": True,
            "intended_faces": len(intended_locations),
            "geometric_faces": len(geometric_locations),
            "matched_faces": len(matched),
            "missing_faces": len(missing),
            "missing_barrier_faces": len(missing_barrier),
            "unexpected_faces": len(unexpected),
            "intended_components": intended_stats["components"],
            "intended_branch_nodes": intended_stats["branch_nodes"],
            "geometric_components": geometric_stats["components"],
            "geometric_branch_nodes": geometric_stats["branch_nodes"],
            "fault_blocks": self._fault_block_count(intended),
            "missing_locations": sorted(missing),
            "unexpected_locations": sorted(unexpected),
        }
        summary["status"] = (
            "PASS" if summary["unexpected_faces"] == 0
            and summary["missing_barrier_faces"] == 0
            and summary["intended_branch_nodes"] == 0
            else "CHECK"
        )
        self._fault_topology_summary = summary
        return summary

    def _write_grid_qc_report(self, grdecl_filename):
        if not bool(getattr(self.model, "write_grid_qc_report", True)):
            return None
        path = Path(grdecl_filename)
        report_path = path.with_name(path.stem + "_GRID_QC.txt")
        dets = self._slanted_cell_center_jacobians()
        pinched = self._pinched_cell_mask()
        active_finite = np.isfinite(dets) & ~pinched
        finite = dets[active_finite]
        median = float(np.median(finite)) if len(finite) else 0.0
        orientation = 1.0 if median >= 0.0 else -1.0
        signed = orientation * dets
        median_abs = max(abs(median), 1.0e-30)
        ratios = signed / median_abs
        bad = np.argwhere(
            ~np.isfinite(ratios)
            | ((ratios <= 0.0) & ~pinched)
        )
        warning_limit = float(getattr(self.model, "grid_qc_warning_ratio", 0.10))
        warnings = np.argwhere(
            np.isfinite(ratios)
            & ~pinched
            & (ratios > 0.0)
            & (ratios < warning_limit)
        )
        records = list(getattr(self, "_coord_alignment_records", []))
        max_cells = max(1, int(getattr(self.model, "grid_qc_max_cells", 20)))
        flat_order = np.argsort(
            np.where(pinched, np.inf, np.where(np.isfinite(ratios), ratios, -np.inf)),
            axis=None,
        )

        with report_path.open("w", encoding="utf-8") as f:
            f.write("PyGRID Grid Quality Report\n")
            f.write("==========================\n\n")
            f.write(f"Version        : {PYGRID_VERSION}\n")
            f.write(f"Build          : {PYGRID_BUILD}\n")
            f.write(f"Fault geometry : {FAULT_GEOMETRY_REVISION}\n")
            f.write(f"Grid           : {path.name}\n")
            f.write(
                f"Dimensions     : {self.model.nx} x {self.model.ny} x "
                f"{self.model.layers}\n\n"
            )

            f.write("FAULT COORD ALIGNMENT\n")
            f.write("---------------------\n")
            summary = dict(getattr(self, "_coord_alignment_summary", {}))
            checked = int(summary.get(
                "checked",
                sum(
                    len(v) for v in self._fault_chain_anchors(
                        self._base_pillar_array()
                    ).values()
                ),
            ))
            primary_detected = int(summary.get(
                "primary_detected",
                sum(1 for r in records if r.get("phase", "PRIMARY") == "PRIMARY"),
            ))
            primary_accepted = int(summary.get(
                "primary_accepted",
                sum(
                    1 for r in records
                    if r.get("phase", "PRIMARY") == "PRIMARY" and r["accepted"]
                ),
            ))
            polish_attempted = int(summary.get(
                "polish_attempted",
                sum(1 for r in records if r.get("phase") == "POLISH"),
            ))
            polish_accepted = int(summary.get(
                "polish_accepted",
                sum(
                    1 for r in records
                    if r.get("phase") == "POLISH" and r["accepted"]
                ),
            ))
            top_polish_attempted = int(summary.get(
                "top_polish_attempted",
                sum(1 for r in records if r.get("phase") == "TOP_POLISH"),
            ))
            top_polish_accepted = int(summary.get(
                "top_polish_accepted",
                sum(
                    1 for r in records
                    if r.get("phase") == "TOP_POLISH" and r["accepted"]
                ),
            ))
            f.write(f"Fault-edge pillars checked : {checked}\n")
            f.write(f"Primary outliers detected   : {primary_detected}\n")
            f.write(f"Primary corrections accepted: {primary_accepted}\n")
            f.write(f"Ribbon polish attempts      : {polish_attempted}\n")
            f.write(f"Ribbon polish accepted      : {polish_accepted}\n")
            f.write(f"Upper-edge polish attempts  : {top_polish_attempted}\n")
            f.write(f"Upper-edge polish accepted  : {top_polish_accepted}\n")
            if summary:
                f.write(
                    f"Remaining above tolerance   : "
                    f"{int(summary.get('remaining_above_tolerance', 0))}\n"
                )
                f.write(
                    f"Final maximum ribbon mismatch: "
                    f"{float(summary.get('final_max_deviation', 0.0)):.3f}\n"
                )
                f.write(
                    f"Upper-edge remaining > tol  : "
                    f"{int(summary.get('top_remaining_above_tolerance', 0))}\n"
                )
                f.write(
                    f"Final maximum upper mismatch: "
                    f"{float(summary.get('top_final_max_deviation', 0.0)):.3f}\n"
                )
            f.write(
                "Tolerance                  : "
                f"{float(getattr(self.model, 'fault_coord_alignment_tolerance_fraction', 0.15)):.3f} "
                "x representative grid spacing\n"
            )
            f.write(
                "Ribbon polish step         : "
                f"{float(getattr(self.model, 'fault_coord_polish_fraction', 0.80)):.2f} "
                "x residual, capped at "
                f"{float(getattr(self.model, 'fault_coord_polish_max_shift_fraction', 0.12)):.3f} "
                "x grid spacing\n"
            )
            f.write("Upper-edge target          : nearest XY point on matched FLT\n")
            f.write("Lower-edge direct FLB snap : OFF - V3.10 stable keeps lower edge as pivot\n")
            f.write(
                "Upper-edge polish step     : "
                f"{float(getattr(self.model, 'fault_coord_top_polish_fraction', 0.80)):.2f} "
                "x residual, capped at "
                f"{float(getattr(self.model, 'fault_coord_top_polish_max_shift_fraction', 0.04)):.3f} "
                "x grid spacing; lower edge is the pivot\n"
            )
            f.write(
                "Upper-edge Jacobian floor  : "
                f"{float(getattr(self.model, 'fault_coord_top_min_jacobian_ratio', 0.10)):.3f}\n"
            )
            f.write(
                "Upper-edge adaptive forcing: "
                f"{'ON' if bool(getattr(self.model, 'fault_coord_top_adaptive', True)) else 'OFF'} "
                f"({int(getattr(self.model, 'fault_coord_top_line_search_steps', 12))} search steps)\n"
            )
            f.write(
                "Upper-edge scan blocked    : "
                f"{'ON' if bool(getattr(self.model, 'fault_coord_top_scan_all', True)) else 'OFF'}\n"
            )
            f.write(
                "Upper-edge continuity guard: "
                f"{'ON' if bool(getattr(self.model, 'fault_coord_top_continuity_guard', True)) else 'OFF'}\n\n"
            )
            if records:
                for r in records:
                    j, i = r["key"]
                    phase = r.get("phase", "PRIMARY")
                    pass_no = int(r.get("pass", 0))
                    status = "CORRECTED" if r["accepted"] else "REJECTED"
                    f.write(
                        f"{status:9s} phase={phase} pass={pass_no} "
                        f"fault={r['name']} pillar=(I={i+1},J={j+1}) "
                        f"station={r['station']:.4f}\n"
                    )
                    f.write(
                        f"  deviation before : {r['deviation_before']:.3f}\n"
                        f"  applied shift    : {r['applied_shift']:.3f}\n"
                        f"  deviation after  : {r['deviation_after']:.3f}\n"
                        f"  trial min Jacobian ratio : "
                        f"{r['trial_min_jacobian_ratio']:.4f}\n"
                    )
                    if phase == "TOP_POLISH":
                        f.write(
                            f"  candidate rank   : {int(r.get('candidate_rank', 1))}\n"
                        )
                        if r.get("adaptive"):
                            f.write(
                                f"  adaptive forcing : YES "
                                f"(full requested shift {r.get('requested_shift', 0.0):.3f})\n"
                            )
                    if phase in {"POLISH", "TOP_POLISH"}:
                        label = "upper max mismatch" if phase == "TOP_POLISH" else "max mismatch"
                        f.write(
                            f"  {label:16s}: "
                            f"{r.get('max_deviation_before', 0.0):.3f} -> "
                            f"{r.get('max_deviation_after', 0.0):.3f}\n"
                        )
            else:
                f.write("No local COORD outliers required correction.\n")

            f.write("\nV4 FAULT PATH OPTIMIZATION\n")
            f.write("--------------------------\n")
            path_summary = dict(getattr(
                self, "_fault_path_optimization_summary", {}
            ))
            if path_summary.get("enabled"):
                f.write(
                    f"Legacy v3.22 faces          : "
                    f"{int(path_summary.get('legacy_faces', 0))}\n"
                )
                f.write(
                    f"Selected v4 faces           : "
                    f"{int(path_summary.get('optimized_faces', 0))}\n"
                )
                f.write(
                    f"Fault components searched   : "
                    f"{int(path_summary.get('components', 0))}\n"
                )
                f.write(
                    f"Components with changed path: "
                    f"{int(path_summary.get('changed_components', 0))}\n"
                )
                f.write(
                    f"Components using fallback   : "
                    f"{int(path_summary.get('fallback_components', 0))}\n"
                )
                f.write(
                    f"Non-conforming components kept v3.22: "
                    f"{int(path_summary.get('preserved_components', 0))}\n"
                )
                f.write(
                    f"Path objective              : "
                    f"{float(path_summary.get('legacy_cost', 0.0)):.6f} -> "
                    f"{float(path_summary.get('optimized_cost', 0.0)):.6f}\n"
                )
                f.write(
                    "Safety rule                 : same endpoints; monotonic connected "
                    "edge path only; v3.22 fallback otherwise\n"
                )
            else:
                f.write("V4 fault-path optimization is disabled.\n")

            f.write("\nV4.1 FAULT-BAND REDISTRIBUTION\n")
            f.write("--------------------------------\n")
            relax = dict(getattr(
                self, "_fault_band_relaxation_summary", {}
            ))
            if relax.get("enabled"):
                f.write(
                    f"Accepted harmonic fraction    : "
                    f"{float(relax.get('accepted_fraction', 0.0)):.3f}\n"
                )
                f.write(
                    f"Non-anchor pillars adjusted   : "
                    f"{int(relax.get('moved_pillars', 0))}\n"
                )
                f.write(
                    f"Displacement roughness        : "
                    f"{float(relax.get('before_roughness', 0.0)):.6g} -> "
                    f"{float(relax.get('after_roughness', 0.0)):.6g}\n"
                )
                f.write(
                    f"Minimum centre-Jacobian ratio : "
                    f"{float(relax.get('min_jacobian_ratio', 0.0)):.4f}\n"
                )
                if 'median_retained_slant' in relax:
                    f.write(
                        f"Median retained fault slant   : "
                        f"{float(relax.get('median_retained_slant', 0.0)):.3f}\n"
                    )
                f.write(
                    f"Logical influence half-width  : "
                    f"{int(relax.get('lines', getattr(self.model, 'fault_band_relax_lines', 6)))} pillar lines\n"
                )
                f.write(
                    "Safety rule                    : FLT anchors frozen; outer footprint fixed; "
                    "95% of accepted Jacobian margin retained\n"
                )
            else:
                f.write("V4.1 harmonic fault-band redistribution is disabled.\n")

            f.write("\nV4 FAULT TOPOLOGY RECONSTRUCTION\n")
            f.write("--------------------------------\n")
            topology = self._fault_topology_qc_summary()
            if topology.get("enabled"):
                f.write(f"Status                       : {topology.get('status', 'CHECK')}\n")
                f.write(
                    f"Intended logical faces       : "
                    f"{int(topology.get('intended_faces', 0))}\n"
                )
                f.write(
                    f"Reconstructed ZCORN faces    : "
                    f"{int(topology.get('geometric_faces', 0))}\n"
                )
                f.write(
                    f"Matching faces               : "
                    f"{int(topology.get('matched_faces', 0))}\n"
                )
                f.write(
                    f"Missing intended faces       : "
                    f"{int(topology.get('missing_faces', 0))}\n"
                )
                f.write(
                    f"Missing barrier faces        : "
                    f"{int(topology.get('missing_barrier_faces', 0))}\n"
                )
                f.write(
                    f"Unexpected geometric faces   : "
                    f"{int(topology.get('unexpected_faces', 0))}\n"
                )
                f.write(
                    f"Intended components/branches : "
                    f"{int(topology.get('intended_components', 0))} / "
                    f"{int(topology.get('intended_branch_nodes', 0))}\n"
                )
                f.write(
                    f"Geometry components/branches : "
                    f"{int(topology.get('geometric_components', 0))} / "
                    f"{int(topology.get('geometric_branch_nodes', 0))}\n"
                )
                f.write(
                    f"Connected I/J fault blocks   : "
                    f"{int(topology.get('fault_blocks', 0))}\n"
                )
                missing = list(topology.get("missing_locations", ()))
                unexpected = list(topology.get("unexpected_locations", ()))
                if missing:
                    f.write("Missing face locations       : " + ", ".join(
                        f"I={i},J={j},{face}" for i, j, face in missing[:20]
                    ) + (" ..." if len(missing) > 20 else "") + "\n")
                if unexpected:
                    f.write("Unexpected face locations    : " + ", ".join(
                        f"I={i},J={j},{face}" for i, j, face in unexpected[:20]
                    ) + (" ..." if len(unexpected) > 20 else "") + "\n")
            else:
                f.write("V4 fault-topology reconstruction is disabled.\n")

            f.write("\nFLT TRACE CONFORMING\n")
            f.write("--------------------\n")
            trace_summary = dict(getattr(
                self, "_fault_trace_conforming_summary", {}
            ))
            if trace_summary:
                f.write(
                    f"Fault-edge anchors targeted    : "
                    f"{int(trace_summary.get('anchors', 0))}\n"
                )
                f.write(
                    f"Tapered deformation lines     : "
                    f"{int(trace_summary.get('lines', 0))} per side\n"
                )
                f.write(
                    f"COORD pillars moved           : "
                    f"{int(trace_summary.get('band_pillars', 0))}\n"
                )
                f.write(
                    f"Collective move fraction      : "
                    f"{float(trace_summary.get('accepted_fraction', 0.0)):.3f}\n"
                )
                f.write(
                    f"Top-edge maximum mismatch     : "
                    f"{float(trace_summary.get('before_max', 0.0)):.3f} -> "
                    f"{float(trace_summary.get('after_max', 0.0)):.3f}\n"
                )
                f.write(
                    f"Top-edge RMS mismatch         : "
                    f"{float(trace_summary.get('before_rms', 0.0)):.3f} -> "
                    f"{float(trace_summary.get('after_rms', 0.0)):.3f}\n"
                )
                f.write(
                    f"Median retained FLT->FLB slant: "
                    f"{float(trace_summary.get('median_retained_slant', 0.0)):.3f}\n"
                )
                f.write(
                    f"Trace-conform Jacobian ratio  : "
                    f"{float(trace_summary.get('min_jacobian_ratio', 0.0)):.4f}\n"
                )
                f.write(
                    "Taper target                  : fault edge 100%, then "
                    "80/60/40/20% over four neighbouring lines\n"
                )
            else:
                f.write("FLT-trace conforming not active for this grid.\n")

            f.write("\nFAULT PLANE STRAIGHTENING\n")
            f.write("---------------------------\n")
            plane_summary = dict(getattr(
                self, "_fault_plane_straightening_summary", {}
            ))
            if plane_summary:
                f.write(
                    f"Fault-edge generators regularised: "
                    f"{int(plane_summary.get('pillars', 0))}\n"
                )
                f.write(
                    f"Collective move fraction       : "
                    f"{float(plane_summary.get('accepted_fraction', 0.0)):.3f}\n"
                )
                f.write(
                    f"Maximum retained-slant jump    : "
                    f"{float(plane_summary.get('max_retention_jump', 0.0)):.3f}\n"
                )
                f.write(
                    f"Straightening Jacobian ratio   : "
                    f"{float(plane_summary.get('min_jacobian_ratio', 0.0)):.4f}\n"
                )
                f.write(
                    "Plane target                   : continuous FLT->FLB "
                    "ruled generator\n"
                )
                f.write(
                    "Safety rule                    : reduce excessive local "
                    "slant only; never increase a guarded pillar\n"
                )
            else:
                f.write("Fault-plane straightening not active for this grid.\n")

            f.write("\nCELL GEOMETRY QC\n")
            f.write("----------------\n")
            f.write(f"Total cells                  : {ratios.size}\n")
            f.write(f"Inverted/non-finite cells    : {len(bad)}\n")
            f.write(f"Pinched/collapsed cells      : {int(np.count_nonzero(pinched))}\n")
            f.write(
                f"Warning cells (< {warning_limit:.3f}) : {len(warnings)}\n"
            )
            active_ratios = ratios[~pinched & np.isfinite(ratios)]
            min_active_ratio = float(np.min(active_ratios)) if active_ratios.size else 0.0
            f.write(f"Minimum active-cell Jacobian ratio: {min_active_ratio:.4f}\n")
            f.write(f"Median centre Jacobian       : {median:.6g}\n")

            f.write("\nWorst cells (lowest normalized centre Jacobian)\n")
            f.write("-----------------------------------------------\n")
            count = 0
            for flat in flat_order:
                k, j, i = np.unravel_index(int(flat), ratios.shape)
                ratio = float(ratios[k, j, i])
                if not np.isfinite(ratio):
                    label = "ERROR"
                elif ratio <= 0.0:
                    label = "ERROR"
                elif ratio < warning_limit:
                    label = "WARNING"
                else:
                    label = "OK"
                f.write(
                    f"{label:7s} I={i+1} J={j+1} K={k+1} "
                    f"ratio={ratio:.4f}\n"
                )
                count += 1
                if count >= max_cells:
                    break

        print(f"PyGRID: grid QC report written: {report_path}")
        return report_path

    def write(self, filename=None, specgrid=True, coord=True, zcorn=True):
        if filename is None:
            filename = self.model.name + ".GRDECL"

        if coord and zcorn:
            self._validate_slanted_cell_geometry()

        with open(filename, "w", encoding="utf-8") as f:
            f.write("-- Generated by PyGRID\n")
            f.write(f"-- PyGRID {PYGRID_VERSION}\n")
            f.write(f"-- Build: {PYGRID_BUILD}\n")
            f.write(f"-- Fault geometry: {FAULT_GEOMETRY_REVISION}\n")

            if not self.model.split_faults:
                f.write("-- Split Grid is not selected.\n")
                f.write("-- Digitized faults are NOT used for interpolation or grid geometry.\n")
                if getattr(self.model, "top_faults", None):
                    f.write("-- A separate FAULTS file may still be written from the TOP fault\n")
                    f.write("-- traces for transmissibility testing.\n")
            elif self._slanted_fault_mode() == "LAYERED":
                f.write("-- ZigZag (stair-step) fault method selected.\n")
                f.write("-- No exported bottom traces are active; faults are carried\n")
                f.write("-- vertically through K from the TOP traces (*.flt).\n")
                f.write("-- Fault faces are hard interpolation/smoothing boundaries except\n")
                f.write("-- where PyGRID detects a zero-throw surface-fault piece.\n")
            elif self._slanted_fault_mode() == "MIXED":
                f.write("-- ZigZag selected with per-fault conforming/slanted overrides.\n")
                f.write("-- TOP-only faults retain classic vertical zigzag geometry.\n")
                f.write("-- Faults having same-name exported BOTTOM traces conform to FLT\n")
                f.write("-- and slant toward FLB using COORD pillars.\n")
                f.write("-- Fault faces remain hard interpolation/smoothing boundaries except\n")
                f.write("-- where PyGRID detects a zero-throw surface-fault piece.\n")
            else:
                f.write("-- Conforming fault geometry selected (ZigZag not selected).\n")
                f.write("-- Every TOP fault (*.flt) is conformed to its continuous trace.\n")
                if (
                    hasattr(self.model, "bottom_faults_defined")
                    and self.model.bottom_faults_defined()
                ):
                    f.write("-- Matching bottom fault traces (*.flb) are honored where defined.\n")
                else:
                    f.write("-- No bottom fault traces (*.flb) were found; unmatched pieces\n")
                    f.write("-- remain vertical.\n")
                f.write("-- Fault faces are hard interpolation/smoothing boundaries except\n")
                f.write("-- where PyGRID detects a zero-throw surface-fault piece.\n")

            f.write("\n")

            # A complete Eclipse grid header is required when geometry is
            # exported in local GRIDUNIT coordinates.  MAPUNITS/MAPAXES retain
            # the global map position independently of the reservoir unit system.
            if specgrid or coord or zcorn:
                self.write_map_header(f)
            if specgrid:
                self.write_specgrid(f)
            if coord:
                self.write_coord(f)
            if zcorn:
                self.write_zcorn(f)

        if coord and zcorn:
            self._write_grid_qc_report(filename)
