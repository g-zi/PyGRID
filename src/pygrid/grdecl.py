"""Eclipse GRDECL output."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .faults import segment_intersection
from .geometry import create_pillars, read_extent


class GRDECLWriter:

    PILLAR_Z_TOP = 0.0
    PILLAR_Z_BOTTOM = 100000.0
    FAULT_SIDE_NUDGE = 1.0e-3

    def __init__(self, model):
        self.model = model
        self._fault_faces_cache = None

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

    def _pillar_array(self):
        return np.asarray(self._pillars(), dtype=float).reshape(
            (self.model.ny + 1, self.model.nx + 1, 2)
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

        values = []
        for trace in self.model.fault_set.named_traces(name, bottom=False):
            for point in trace.points:
                point = np.asarray(point, dtype=float)
                displacement = self.model.fault_set.slant_displacement(name, point)
                bottom_xy = point + displacement
                z_top = float(top_map.interpolate(point))
                z_bottom = float(bottom_map.interpolate(bottom_xy))
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

        z_top = float(top_map.interpolate(np.asarray(top_xy, dtype=float)))

        if bottom_map is not None:
            z_bottom = float(
                bottom_map.interpolate(np.asarray(bottom_xy, dtype=float))
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

    def _coord_pillars(self):
        """Return per-pillar COORD endpoints, including slanted fault sticks.

        The normal grid remains unchanged.  Only pillars that form a fault face
        whose name occurs in both ``*.flt`` and ``*.flb`` are inclined.

        For such a pillar, the top staircase position is kept fixed and the
        local FLT->FLB displacement is applied at the structural bottom depth.
        Two nearby points on that same line are written to COORD, with a modest
        depth margin so the generated cells lie safely inside the pillar span.
        """
        base = self._pillar_array()
        ny, nx = self.model.ny, self.model.nx

        coord_top = np.empty((ny + 1, nx + 1, 3), dtype=float)
        coord_bottom = np.empty((ny + 1, nx + 1, 3), dtype=float)

        coord_top[:, :, :2] = base
        coord_top[:, :, 2] = self.PILLAR_Z_TOP
        coord_bottom[:, :, :2] = base
        coord_bottom[:, :, 2] = self.PILLAR_Z_BOTTOM

        fault_pillars = self._fault_pillar_names()
        if not fault_pillars:
            return coord_top, coord_bottom

        for (j, i), names in fault_pillars.items():
            base_xy = base[j, i]
            top_candidates = []
            bottom_candidates = []

            for name in sorted(names):
                displacement = self.model.fault_set.slant_displacement(
                    name, base_xy
                )
                if float(np.linalg.norm(displacement)) <= 1.0e-9:
                    continue

                desired_bottom_xy = base_xy + displacement
                depths = self._fault_reference_depths(
                    name, base_xy, desired_bottom_xy
                )

                if depths is None:
                    # No structural depth information: retain the historical
                    # Z endpoints but still encode the requested lateral tilt.
                    top_candidates.append(
                        np.asarray(
                            [base_xy[0], base_xy[1], self.PILLAR_Z_TOP],
                            dtype=float,
                        )
                    )
                    bottom_candidates.append(
                        np.asarray(
                            [
                                desired_bottom_xy[0],
                                desired_bottom_xy[1],
                                self.PILLAR_Z_BOTTOM,
                            ],
                            dtype=float,
                        )
                    )
                    continue

                z_fault_top, z_fault_bottom = depths
                dz = z_fault_bottom - z_fault_top
                slope = displacement / dz

                # Define a short, well-conditioned pillar segment that brackets
                # both interpreted fault traces.  COORD only needs two points
                # on the straight pillar; they need not use global 0/100000 Z.
                margin = max(25.0, 0.20 * abs(dz))
                z1 = min(z_fault_top, z_fault_bottom) - margin
                z2 = max(z_fault_top, z_fault_bottom) + margin

                xy1 = base_xy + slope * (z1 - z_fault_top)
                xy2 = base_xy + slope * (z2 - z_fault_top)

                top_candidates.append(
                    np.asarray([xy1[0], xy1[1], z1], dtype=float)
                )
                bottom_candidates.append(
                    np.asarray([xy2[0], xy2[1], z2], dtype=float)
                )

            if top_candidates:
                # At a rare intersection of two slanted faults, average the
                # independently defined sticks rather than choosing one name.
                coord_top[j, i] = np.mean(top_candidates, axis=0)
                coord_bottom[j, i] = np.mean(bottom_candidates, axis=0)

        return coord_top, coord_bottom

    def write_coord(self, f):
        coord_top, coord_bottom = self._coord_pillars()

        f.write("COORD\n")
        for j in range(self.model.ny + 1):
            for i in range(self.model.nx + 1):
                xt, yt, zt = coord_top[j, i]
                xb, yb, zb = coord_bottom[j, i]
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

    def _interpolate(self, map_object, xy, barrier_origins=None):
        if self.model.split_faults and self.model.fault_set:
            return map_object.interpolate_faulted(
                xy,
                self.model.fault_set,
                search_radius=self.model.search_radius,
                barrier_origins=barrier_origins,
            )
        return map_object.interpolate(xy)

    def _cell_corner_queries(self):
        """
        Return four XY queries per cell, each slightly inside that cell.

        Adjacent cells therefore query opposite sides of a fault independently.
        This is the key that lets ZCORN contain a genuine depth discontinuity
        while COORD can remain a simple vertical-pillar grid.

        Corner order is NW, NE, SW, SE in the grid-array convention.
        """
        p = self._pillar_array()
        ny, nx = self.model.ny, self.model.nx
        q = np.empty((ny, nx, 4, 2), dtype=float)
        n = self.FAULT_SIDE_NUDGE

        for j in range(ny):
            for i in range(nx):
                corners = np.asarray(
                    [p[j, i], p[j, i + 1], p[j + 1, i], p[j + 1, i + 1]],
                    dtype=float,
                )
                center = corners.mean(axis=0)
                q[j, i] = corners + n * (center - corners)

        return q

    def _fault_face_sets(self):
        """Return zero-based X+/Y+ fault-face lookup sets."""
        x_faces = set()
        y_faces = set()
        for _name, i1, j1, face in self.fault_faces():
            key = (i1 - 1, j1 - 1)
            if face == "X+":
                x_faces.add(key)
            elif face == "Y+":
                y_faces.add(key)
        return x_faces, y_faces

    def _split_corner_topology(self):
        """
        Build split pillar nodes from the v3 fault-face topology.

        Every logical cell owns four ZCORN corner values.  Corners at the same
        physical pillar are merged only across *non-faulted* shared faces.
        Across a fault they remain separate nodes, so subsequent smoothing can
        never average the two sides together.
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

        x_faults, y_faults = self._fault_face_sets()

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

    def _smooth_surface(self, surface):
        """Smooth one already fault-split structural surface.

        The split-node topology is created from the frozen v3 fault faces.
        Duplicate cell corners are merged only where cells are genuinely
        connected.  Therefore every smoothing method operates independently
        on the two sides of a fault and cannot reduce the fault throw by
        averaging through the fault face.
        """
        iterations = int(getattr(self.model, "smooth_iterations", 0) or 0)
        factor = float(getattr(self.model, "smooth_factor", 0.15) or 0.0)

        groups, neighbours = self._split_corner_topology()
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

    def _connected_corner_average(self, values):
        """Make duplicate corners identical without spatial smoothing.

        This is useful for thickness maps: a thickness value may vary in space,
        but corners belonging to one ordinary (non-faulted) pillar must agree.
        Across a fault the split corners remain independent.
        """
        groups, _ = self._split_corner_topology()
        nodes = self._group_average(values, groups)
        return nodes[groups]

    def _query_map_at_cell_corners(self, map_object, flat, barrier_origins):
        values = np.asarray(
            self._interpolate(
                map_object, flat, barrier_origins=barrier_origins
            ),
            dtype=float,
        )
        return values.reshape((self.model.ny, self.model.nx, 4))

    def _cell_corner_interfaces(self):
        if not hasattr(self.model, "top_map"):
            raise ValueError("model.top must be set before writing ZCORN")

        ny, nx, nz = self.model.ny, self.model.nx, self.model.layers
        queries = self._cell_corner_queries()
        flat = queries.reshape((-1, 2))

        # Use the cell centre to decide which side of every digitized fault
        # the complete cell belongs to.  All structural surfaces use the same
        # barrier origins and therefore the same frozen v3 fault topology.
        centres = queries.mean(axis=2)
        barrier_origins = np.repeat(
            centres[:, :, None, :], 4, axis=2
        ).reshape((-1, 2))

        top = self._query_map_at_cell_corners(
            self.model.top_map, flat, barrier_origins
        )
        top = self._smooth_surface(top)

        mode = str(getattr(self.model, "layer_mode", "") or "").upper()
        fractions = self._layer_fractions()

        # ------------------------------------------------------------
        # Structural TOP + structural BOTTOM.
        # Layer thicknesses are relative proportions between the two
        # independently interpreted and independently smoothed horizons.
        #
        # Optional thickness-derived internal horizon:
        #     MID = TOP + THICKNESS
        # at model.thickness_split_layer.  The proportional weights are then
        # normalized independently above and below MID, exactly like TinyECL's
        # current tNavigator workflow.
        # ------------------------------------------------------------
        if mode in {"PROPORTIONAL", "BOTTOM"} and hasattr(
            self.model, "bottom_map"
        ):
            bottom = self._query_map_at_cell_corners(
                self.model.bottom_map, flat, barrier_origins
            )
            bottom = self._smooth_surface(bottom)

            minimum_zone_thickness = float(
                getattr(self.model, "minimum_zone_thickness", 0.0) or 0.0
            )
            if minimum_zone_thickness < 0.0:
                raise ValueError("model.minimum_zone_thickness must be >= 0")

            # First make TOP/BOTTOM structurally valid.
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
                        self.model.thickness_map, flat, barrier_origins
                    )
                    thickness = self._connected_corner_average(thickness)
                    if np.any(thickness <= 0.0):
                        raise ValueError(
                            "Interpolated thickness must be positive everywhere; "
                            f"minimum is {float(np.min(thickness))}"
                        )

                    # Create the structural MID surface from TOP + THICKNESS,
                    # then apply the same fault-connectivity-aware smoothing
                    # used for TOP and BOTTOM.
                    mid = self._smooth_surface(top + thickness)
                else:
                    raise ValueError(
                        "model.thickness_mode currently supports only "
                        "'FROM_TOP' for an internal thickness horizon"
                    )

                # MID must lie between TOP and BOTTOM.  Clip only where the
                # independent interpretations would otherwise invert a zone.
                lower_limit = top + minimum_zone_thickness
                upper_limit = bottom - minimum_zone_thickness
                invalid_zone = upper_limit < lower_limit
                if np.any(invalid_zone):
                    # With minimum thickness zero this is already handled by
                    # the TOP/BOTTOM pinch-out above.  For positive minima,
                    # fall back to the midpoint where both minima cannot fit.
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

                # Layers 1..split_layer between TOP and MID.
                upper_sep = mid - top
                for k in range(split_layer + 1):
                    interfaces[k] = top + upper_sep * upper_frac[k]

                # Layers split_layer+1..nz between MID and BOTTOM.
                lower_sep = bottom - mid
                for local_k in range(1, len(lower_frac)):
                    k = split_layer + local_k
                    interfaces[k] = mid + lower_sep * lower_frac[local_k]

                return interfaces

            # No internal thickness horizon: one proportional zone TOP->BOTTOM.
            interfaces = np.empty((nz + 1, ny, nx, 4), dtype=float)
            for k, fraction in enumerate(fractions):
                interfaces[k] = top + separation * fraction
            return interfaces

        # ------------------------------------------------------------
        # TOP + THICKNESS map.  The thickness map defines the total
        # vertical separation.  It is not spatially smoothed here; only
        # genuinely connected duplicate corners are reconciled.
        # ------------------------------------------------------------
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
                self.model.thickness_map, flat, barrier_origins
            )
            thickness = self._connected_corner_average(thickness)
            if np.any(thickness <= 0.0):
                raise ValueError(
                    "Interpolated thickness must be positive everywhere; "
                    f"minimum is {float(np.min(thickness))}"
                )

            interfaces = np.empty((nz + 1, ny, nx, 4), dtype=float)
            for k, fraction in enumerate(fractions):
                interfaces[k] = top + thickness * fraction
            return interfaces

        # ------------------------------------------------------------
        # Layer-cake fallback.  With no bottom/thickness structure,
        # layer_thicknesses are absolute DZ values in model.depth_units.
        # ------------------------------------------------------------
        if mode in {"ABSOLUTE", "DZ"}:
            thicknesses = self.model.layer_thicknesses
            if thicknesses is None:
                raise ValueError(
                    "Absolute layer-cake mode requires "
                    "model.layer_thicknesses"
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
            return interfaces

        # Backwards-compatible fallback for older RUNFILE.PY files:
        # if a thickness map exists, retain the original top+thickness
        # proportional behaviour.
        if hasattr(self.model, "thickness_map"):
            thickness = self._query_map_at_cell_corners(
                self.model.thickness_map, flat, barrier_origins
            )
            thickness = self._connected_corner_average(thickness)
            if np.any(thickness <= 0.0):
                raise ValueError(
                    "Interpolated thickness must be positive everywhere; "
                    f"minimum is {float(np.min(thickness))}"
                )
            interfaces = np.empty((nz + 1, ny, nx, 4), dtype=float)
            for k, fraction in enumerate(fractions):
                interfaces[k] = top + thickness * fraction
            return interfaces

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
        """Return XY centres for all logical grid cells."""
        p = self._pillar_array()
        return 0.25 * (
            p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:]
        )

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

        centres = self._cell_centres()
        nx, ny = self.model.nx, self.model.ny
        found = set()

        for trace in self.model.fault_set.traces:
            for f1, f2 in trace.segments:
                # Adjacent I cells: split their common X face.
                for j in range(ny):
                    for i in range(nx - 1):
                        if segment_intersection(
                            centres[j, i], centres[j, i + 1], f1, f2
                        ) is not None:
                            found.add((trace.name, i + 1, j + 1, "X+"))

                # Adjacent J cells: split their common Y face.
                for j in range(ny - 1):
                    for i in range(nx):
                        if segment_intersection(
                            centres[j, i], centres[j + 1, i], f1, f2
                        ) is not None:
                            found.add((trace.name, i + 1, j + 1, "Y+"))

        self._fault_faces_cache = sorted(
            found, key=lambda x: (x[0], x[2], x[1], x[3])
        )
        return self._fault_faces_cache

    def write(self, filename=None, specgrid=True, coord=True, zcorn=True):
        if filename is None:
            filename = self.model.name + ".GRDECL"

        with open(filename, "w", encoding="utf-8") as f:
            f.write("-- Generated by PyGRID\n")
            if self.model.split_faults and self.model.fault_set:
                f.write("-- Fault-barrier interpolation and split ZCORN enabled\n\n")
            else:
                f.write("-- Basic unfaulted grid\n\n")

            if specgrid:
                self.write_specgrid(f)
            if coord:
                self.write_coord(f)
            if zcorn:
                self.write_zcorn(f)
