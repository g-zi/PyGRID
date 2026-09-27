from .faults import FaultSet
from .grdecl import GRDECLWriter
from .maps import ContourMap, ThicknessMap
from .geometry import create_pillars, read_extent
from .license import require_tinyecl_license
from .version import PYGRID_VERSION, PYGRID_BUILD, FAULT_GEOMETRY_REVISION

from pathlib import Path
import numpy as np


class GridModel:

    def __init__(
        self,
        name,
        units,
        xmin,
        ymin,
        xmax,
        ymax,
        nx,
        ny,
    ):
        self.name = name
        self.units = units
        # TinyECL may define horizontal/surface units and depth units separately.
        self.surface_units = units
        self.depth_units = units
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.nx = nx
        self.ny = ny

        self.extent = None
        self.top = None
        self.bottom = None
        self.thickness = None
        # Optional thickness-derived internal structural horizon.
        # FROM_TOP means MID = TOP + THICKNESS.
        self.thickness_mode = None
        self.thickness_split_layer = 0
        self.top_faults = None
        self.bottom_faults = None
        self.wells = None
        self.layer_mode = None
        self.layers = None
        self.layer_thicknesses = None
        # Structural zones may pinch out where interpreted horizons meet/cross.
        # Zero matches TinyECL/tNavigator's current minimum-zone-thickness use.
        self.minimum_zone_thickness = 0.0
        self.angle = 0
        self.rotation_center = None
        # Optional Eclipse MAPAXES override: (x1, y1, x0, y0, x2, y2),
        # expressed in surface_units.  None derives an equivalent local map
        # frame automatically from the TinyECL extent rectangle.
        self.mapaxes = None
        self.split_faults = False
        # GRID-style contour search radius.  TOP and BOTTOM are interpolated
        # independently on their own endpoint geometry before K layers are filled.
        self.search_radius = None
        # Contour lines are interpreted as point clouds.  None means choose a
        # sampling interval automatically from the grid-pillar spacing.
        # Set a positive number for an explicit interval in surface units.
        self.contour_sample_spacing = None
        # Automatic spacing as a fraction of representative grid spacing.
        # 0.35 gives several samples per grid block without excessive size.
        self.contour_sample_fraction = 0.35
        # Automatic zero-throw detection: a vertical top-fault piece crossed
        # by multiple distinct top contour levels remains a FAULTS/grid split,
        # but is transparent to structural-surface interpolation/smoothing.
        self.detect_zero_throw_faults = True
        self.zero_throw_min_contours = 2
        self.zero_throw_min_station_fraction = 0.15

        # Slanted-fault staircase equalisation.  Two grid pillars can belong to
        # the same tiny along-fault station when a stair-stepped X/Y fault face
        # turns a corner.  If one of those two sticks is a strong local depth
        # outlier it creates a narrow triangular spike in the fault plane.
        # Detect only that local outlier and copy the good neighbour's local
        # fault-plane definition; the rest of the slanted fault is untouched.
        self.equalize_slanted_fault_spikes = True
        self.slanted_spike_corner_fraction = 0.25
        self.slanted_spike_error_ratio = 1.75
        self.slanted_spike_relative_jump = 0.20

        # TinyECL fault-method switch.
        #
        # zigzag = True:
        #   TinyECL CreateGrid "ZigZag" is selected.  TOP-only faults keep the
        #   classic stair-step/vertical geometry.  A fault for which TinyECL
        #   also exported a matching BOTTOM trace (*.flb) is an explicit
        #   per-fault request for the conforming/slanted COORD geometry.
        #
        # zigzag = False:
        #   Every TOP fault is conformed to its continuous FLT trace.  Where a
        #   matching BOTTOM trace exists the conformed fault is also slanted by
        #   inclining the relevant COORD pillars; otherwise it remains vertical.
        self.zigzag = True

        # AUTO is the normal TinyECL setting:
        #   ZigZag OFF -> COORD: every TOP fault conforms; FLB adds dip.
        #   ZigZag ON  -> MIXED when FLB exists: only matching FLT/FLB faults
        #                 conform/slant; TOP-only faults remain zigzag.
        #   ZigZag ON with no FLB -> LAYERED: all TOP faults remain zigzag.
        #
        # Explicit LAYERED or COORD is retained as an advanced/backward-
        # compatibility override for hand-written PyGRID run files.
        self.slanted_fault_mode = "AUTO"

        # True-slanted COORD geometry can relax the lower grid across a few
        # neighbouring lines to avoid very narrow cells.  This is independent
        # of the TinyECL ZigZag checkbox.
        self.relax_slanted_faults = True
        self.slanted_fault_relax_lines = 4

        # Optional fault-conforming XY grid.  In COORD mode PyGRID can
        # move every logical TOP fault edge onto FLT and rebuild the crossing logical
        # row/column locally at 90 degrees to the continuous fault tangent.
        # While marching along the fault the transverse logical direction can
        # switch between I and J; NX*NY remains unchanged.
        # V3.12 TinyECL rule: ZigZag OFF means conform the TOP grid to every
        # exported FLT by default.  This flag remains available as an advanced
        # hand-written RUNFILE override; it is ignored in LAYERED/ZigZag mode.
        self.conform_slanted_faults = True
        self.slanted_fault_conform_lines = 4
        # Legacy geometry-safety cap used by the fault-conforming method when
        # the adaptive 3-D COORD topology guard is disabled.  With the guard
        # enabled (normal TinyECL operation), PyGRID preserves the interpreted
        # FLT->FLB slant first and steepens only local pillars that would
        # otherwise invert cells.  This avoids coarse-grid accordion artefacts.
        self.slanted_fault_conform_max_pillar_slope = 0.85
        # V3.17: an exact FLT anchor remains authoritative, but on a sufficiently
        # fine/rotated grid the transverse conforming deformation around it can
        # fold one TOP plan cell.  Repair only the surrounding non-anchor
        # deformation; never pull a fault-edge anchor off the digitized FLT.
        self.slanted_fault_top_topology_guard = True
        self.slanted_fault_top_min_area_ratio = 0.0
        self.slanted_fault_top_repair_factor = 0.85
        self.slanted_fault_top_repair_passes = 64
        # V3.14: on fine grids the same physical FLT->FLB offset spans more
        # cells.  A fixed four-line lower deformation band can then fold the
        # BOTTOM plan-view grid even though the TOP grid remains valid.  Repair
        # only the local lower-grid displacement needed to preserve topology.
        self.slanted_fault_bottom_topology_guard = True
        self.slanted_fault_bottom_min_area_ratio = 0.02
        self.slanted_fault_bottom_repair_factor = 0.90
        self.slanted_fault_bottom_repair_passes = 48
        # V3.16: use a cell-local BOTTOM area floor and allow up to 48 repair passes.
        # A positive BOTTOM plan grid can still produce locally inverted
        # 3-D cells when the straight COORD pillars are too oblique for a fine
        # grid.  Preserve the exact conforming TOP fault trace and steepen only
        # those local fault pillars needed to restore a positive 3-D Jacobian.
        self.slanted_fault_coord_topology_guard = True
        self.slanted_fault_coord_min_jacobian_ratio = 0.005
        self.slanted_fault_coord_repair_factor = 0.90
        self.slanted_fault_coord_repair_passes = 32
        # V3.22: FLT is the authoritative visible top-fault trace.  Correct
        # each fault-edge COORD anchor together with a tapered transverse band
        # of neighbouring pillars (default 100/80/60/40/20 percent) so a
        # row/column switch does not force the green fault line into a kink.
        self.fault_trace_conforming = True
        self.fault_trace_conform_lines = 4
        self.fault_trace_conform_min_jacobian_ratio = 0.02
        # Preserve the v3.20 slant objective while straightening the visible
        # FLT edge: the collective band move is reduced if the median retained
        # FLT->FLB slant would fall below this fraction.
        self.fault_trace_conform_min_median_slant_fraction = 0.75
        self.fault_trace_conform_line_search_steps = 14
        # V3.21 retained-slant regularisation is kept as an optional backwards
        # comparison only; V3.22 does not use it by default.
        self.fault_plane_straightening = False
        self.fault_plane_max_retention_step = 0.15
        self.fault_plane_straightening_min_jacobian_ratio = 0.02
        self.fault_plane_straightening_line_search_steps = 12
        # V3.10: align the actual upper fault-edge COORD intersections directly to FLT.
# V3.8: preserve V3.7 upper-edge polishing and add adaptive step control.
        # V3.7: preserve V3.6 ribbon polishing and add a separate upper-edge
        # correction that pivots each selected stick about its current lower
        # fault-edge point.  This lets the remaining upper fault line move
        # toward the ruled FLT->FLB plane without disturbing the good lower line.
        #
        # Compare each existing fault-edge COORD stick with the smooth
        # local fault ribbon defined by its immediate along-fault neighbours.
        # Clear outliers receive the proven V3.4 correction first.  A short
        # conservative polish then re-measures the resulting ribbon and nudges
        # only the strongest remaining mismatch.  Logical cells/topology are
        # never switched.
        self.align_fault_coord_outliers = True
        self.fault_coord_alignment_tolerance_fraction = 0.12
        # Number of gentle post-V3.4 polish attempts.  Each pass considers only
        # the strongest remaining outlier above the same tolerance.
        self.fault_coord_polish_passes = 4
        # Apply only part of the measured residual mismatch per polish pass.
        self.fault_coord_polish_fraction = 0.80
        # Absolute polish-step cap as a fraction of representative grid spacing.
        self.fault_coord_polish_max_shift_fraction = 0.12
        # Upper-edge-only final polish.  V3.10 measures the actual top fault
        # edge directly against the matched FLT polyline.  Each existing COORD
        # stick is rotated about its good lower-edge pivot toward the nearest
        # point on FLT.  The large nominal shift cap simply means "try the full
        # geometric correction"; the adaptive Jacobian line search remains the
        # hard safety limiter.
        self.fault_coord_top_polish_passes = 80
        self.fault_coord_top_polish_fraction = 1.00
        self.fault_coord_top_tolerance_fraction = 0.01
        self.fault_coord_top_polish_max_shift_fraction = 1.00
        # Upper-edge polish is deliberately stricter than the general COORD
        # repair. V3.8 allows a small QC warning band down to 0.08 so the
        # upper edge can be forced closer; the report still flags ratios < 0.10.
        self.fault_coord_top_min_jacobian_ratio = 0.08
        # If the requested upper-edge step would violate the Jacobian floor,
        # V3.8 searches for the largest smaller step that still passes QC.
        self.fault_coord_top_adaptive = True
        self.fault_coord_top_line_search_steps = 12
        self.fault_coord_top_min_step_fraction = 0.02
        # V3.9: if the strongest upper-edge candidate cannot move safely,
        # continue testing the remaining candidates rather than terminating
        # the whole polishing pass.  A local neighbour-window guard prevents
        # an accepted move from worsening along-fault continuity.
        self.fault_coord_top_scan_all = True
        self.fault_coord_top_continuity_guard = True
        # A correction is accepted only if the complete corner-point grid stays
        # non-inverted and this minimum normalized centre-Jacobian is retained.
        self.fault_coord_alignment_min_jacobian_ratio = 0.02
        # Standard human-readable QC companion file next to the GRDECL.
        self.write_grid_qc_report = True
        self.grid_qc_warning_ratio = 0.10
        self.grid_qc_max_cells = 20
        # Automatically enlarge the cross-fault lower-grid relaxation band
        # when the FLT->FLB movement is large.  The target change between
        # neighbouring bottom-grid lines is at most this fraction of a normal
        # block width.
        self.slanted_fault_max_step_fraction = 0.35
        # True COORD faults are oriented from the ruled FLT->FLB fault plane.
        # Limit only pathological local staircase-corner pillar slopes after
        # converting XY and Z to one physical unit system.  H/V=1.5 corresponds
        # to a fault dip of about 34 degrees from horizontal.  Set <=0 to disable.
        self.slanted_fault_max_pillar_slope = 1.5
        # Reject a true-COORD grid if any cell is inverted at its trilinear
        # centre.  This check is intentionally inactive for ZigZag/vertical mode.
        self.validate_slanted_geometry = True
        # COORD endpoints are written directly on the already-built TOP and
        # BOTTOM structural surfaces; no artificial 0..100000 extension is
        # used for a complete grid.
        self.coord_endpoints_on_surfaces = True
        self.smoothing_method = "LAPLACIAN"
        self.smooth_iterations = 0
        self.smooth_factor = 0.15
        # Taubin's second (inflation) pass.  None means use -1.05 * smooth_factor.
        self.smooth_taubin_mu = None
        # Bilateral Z scale.  None chooses a robust value automatically from
        # connected-neighbour depth differences.
        self.smooth_bilateral_sigma_z = None
        # Biharmonic curvature penalty. None derives it from iterations*factor.
        self.smooth_biharmonic_lambda = None
        self.fault_set = FaultSet()
        self._last_fault_faces = []

    def effective_slanted_fault_mode(self):
        """Return the effective fault-geometry mode for this run.

        AUTO follows the TinyECL CreateGrid ZigZag checkbox:

        * ZigZag OFF -> COORD: every TOP fault conforms; matching FLB adds dip.
        * ZigZag ON + FLB -> MIXED: only faults with a matching FLB conform/slant.
          TOP-only faults keep the classic vertical zigzag geometry.
        * ZigZag ON without FLB -> LAYERED: all TOP faults stay zigzag.

        Explicit LAYERED/MIXED/COORD values remain available for advanced or
        hand-written run files.
        """
        mode = str(self.slanted_fault_mode or "AUTO").strip().upper()
        if mode == "AUTO":
            if not bool(self.zigzag):
                return "COORD"
            return "MIXED" if self.bottom_faults_defined() else "LAYERED"
        if mode not in {"LAYERED", "MIXED", "COORD"}:
            raise ValueError(
                "model.slanted_fault_mode must be 'AUTO', 'LAYERED', "
                "'MIXED' or 'COORD'"
            )
        return mode

    def bottom_faults_defined(self):
        """True when the configured TinyECL bottom-fault file contains data.

        TinyECL may deliberately leave an empty/header-only ``*.flb`` file when
        no bottom faults are selected for export.  Treat that the same as no
        bottom-fault interpretation so ZigZag remains purely layered.
        """
        if not self.bottom_faults:
            return False
        path = Path(self.bottom_faults)
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if not text or text.startswith("*") or text.startswith("--"):
                        continue
                    parts = text.replace(",", " ").split()
                    if len(parts) >= 3:
                        return True
        except OSError:
            return False
        return False

    def rotate(self, angle, x, y):
        self.angle = angle
        self.rotation_center = (x, y)

    def _map_sample_spacing(self):
        """Return contour point-cloud spacing in surface coordinate units."""
        explicit = self.contour_sample_spacing
        if explicit is not None:
            explicit = float(explicit)
            if explicit <= 0.0:
                raise ValueError(
                    "model.contour_sample_spacing must be > 0 or None"
                )
            return explicit

        fraction = float(self.contour_sample_fraction)
        if fraction <= 0.0:
            raise ValueError(
                "model.contour_sample_fraction must be > 0"
            )

        values = []
        if self.extent:
            corners = read_extent(self.extent)
            p = np.asarray(
                create_pillars(corners, self.nx, self.ny), dtype=float
            ).reshape((self.ny + 1, self.nx + 1, 2))
            if p.shape[1] > 1:
                values.extend(
                    np.linalg.norm(p[:, 1:] - p[:, :-1], axis=2).ravel()
                )
            if p.shape[0] > 1:
                values.extend(
                    np.linalg.norm(p[1:] - p[:-1], axis=2).ravel()
                )

        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values) & (values > 1.0e-9)]
        if len(values):
            grid_spacing = float(np.median(values))
        else:
            dx = abs(float(self.xmax) - float(self.xmin)) / max(int(self.nx), 1)
            dy = abs(float(self.ymax) - float(self.ymin)) / max(int(self.ny), 1)
            candidates = [v for v in (dx, dy) if v > 1.0e-9]
            grid_spacing = min(candidates) if candidates else 1.0

        return max(grid_spacing * fraction, 1.0e-9)

    @staticmethod
    def _read_and_sample_map(map_class, filename, spacing):
        map_object = map_class(filename).read()
        before = map_object.source_point_count
        map_object.densify(spacing)
        after = len(map_object.points)
        print(
            f"PyGRID: sampled {filename}: {before} -> {after} points "
            f"(spacing {spacing:g})."
        )
        return map_object

    def read_maps(self):
        spacing = self._map_sample_spacing()

        # Treat several grid-block widths around the contour-data boundary as
        # an extrapolation band.  This catches unsupported Delaunay triangles
        # between terminating contour lines before they can flatten/collapse a
        # structural horizon at the grid edge.  With the default sampling
        # fraction 0.35 this is six representative grid-block widths.
        fraction = max(float(self.contour_sample_fraction), 1.0e-9)
        edge_distance = 6.0 * spacing / fraction

        if self.top:
            self.top_map = self._read_and_sample_map(
                ContourMap, self.top, spacing
            )
            self.top_map.edge_extrapolation_distance = edge_distance
        if self.bottom:
            self.bottom_map = self._read_and_sample_map(
                ContourMap, self.bottom, spacing
            )
            self.bottom_map.edge_extrapolation_distance = edge_distance
        if self.thickness:
            self.thickness_map = self._read_and_sample_map(
                ThicknessMap, self.thickness, spacing
            )
            self.thickness_map.edge_extrapolation_distance = edge_distance

    def read_faults(self):
        """Load only the faults that are allowed to influence grid geometry.

        Split Grid is the master switch.

        * Split Grid OFF: no fault is used by interpolation or grid geometry.
        * Split Grid ON + ZigZag with no exported BOTTOM traces: use TOP
          (*.flt) only; faults remain vertical zigzags through K.
        * Split Grid ON + ZigZag with exported BOTTOM traces: TOP-only faults
          remain vertical zigzags, while same-name TOP/BOTTOM faults conform
          and slant using COORD pillars.
        * Split Grid ON + ZigZag OFF: conform every TOP (*.flt) trace. Matching
          BOTTOM (*.flb) traces additionally make those faults slant with depth;
          TOP faults without FLB remain vertical but conforming.
        """
        if self.split_faults and self.top_faults:
            mode = self.effective_slanted_fault_mode()
            bottom_source = (
                self.bottom_faults
                if mode in {"COORD", "MIXED"}
                else None
            )

            self.fault_set = FaultSet.from_tinyecl(
                self.top_faults,
                bottom_source,
            )

            if (
                self.detect_zero_throw_faults
                and hasattr(self, "top_map")
                and self.fault_set
            ):
                self.fault_set.classify_surface_barriers(
                    self.top_map,
                    min_crossings=self.zero_throw_min_contours,
                    min_station_separation=(
                        self.zero_throw_min_station_fraction
                    ),
                )

                for index, trace in enumerate(self.fault_set.traces, start=1):
                    if trace.surface_barrier:
                        continue
                    levels = ", ".join(
                        f"{z:g}" for z, _station in trace.surface_crossings
                    )
                    print(
                        "PyGRID: zero-throw surface fault: "
                        f"{trace.name} piece {index} "
                        f"(top contours {levels})."
                    )
        else:
            self.fault_set = FaultSet()

    def write_grdecl(self, specgrid=True, coord=True, zcorn=True):
        require_tinyecl_license("write the GRDECL grid")
        self.read_maps()
        self.read_faults()

        writer = GRDECLWriter(self)
        writer.write(specgrid=specgrid, coord=coord, zcorn=zcorn)
        self._last_fault_faces = writer.fault_faces()

    def write_faults(self, filename):
        """Write Eclipse FAULTS independently of whether grid splitting is on.

        This is intentional: with Split Grid OFF the GRDECL geometry remains
        completely unfaulted, but the user may still include the generated
        FAULTS file to test fault transmissibility quickly.
        """
        require_tinyecl_license("write the FAULTS file")
        writer = GRDECLWriter(self)
        nz = int(self.layers or 0)

        if self.split_faults:
            if not self.fault_set:
                self.read_faults()
            records = writer.layered_fault_faces()
        elif self.top_faults and nz > 0:
            # Generate transmissibility faces from TOP traces only without
            # installing them as interpolation barriers or split-grid geometry.
            output_fault_set = FaultSet.from_tinyecl(self.top_faults, None)
            faces = writer.faces_for_fault_set(output_fault_set)
            records = [
                (name, i, j, 1, nz, face)
                for name, i, j, face in faces
            ]
        else:
            records = []

        with open(filename, "w", encoding="utf-8") as f:
            f.write("-- Generated by PyGRID\n")
            f.write(f"-- PyGRID {PYGRID_VERSION}\n")
            f.write(f"-- Build: {PYGRID_BUILD}\n")
            f.write(f"-- Fault geometry: {FAULT_GEOMETRY_REVISION}\n")

            if not self.split_faults:
                f.write("-- NOTE: Split Grid was not selected in TinyECL.\n")
                f.write("-- The GRDECL geometry was generated without honoring faults.\n")
                f.write("-- These FAULTS records are written from the digitized TOP\n")
                f.write("-- fault traces (*.flt) so fault transmissibility can be\n")
                f.write("-- tested separately by including this file in the simulator.\n")
            elif self.effective_slanted_fault_mode() == "LAYERED":
                f.write("-- NOTE: ZigZag (stair-step) fault method selected.\n")
                f.write("-- No exported bottom fault traces were found; all faults are\n")
                f.write("-- carried vertically through K from the TOP traces (*.flt).\n")
            elif self.effective_slanted_fault_mode() == "MIXED":
                f.write("-- NOTE: ZigZag selected with per-fault conforming overrides.\n")
                f.write("-- TOP-only faults keep classic vertical zigzag geometry.\n")
                f.write("-- Same-name TOP/BOTTOM faults are conforming and slanted.\n")
            else:
                f.write("-- NOTE: Conforming fault geometry selected (ZigZag not selected).\n")
                f.write("-- Every TOP fault (*.flt) is conformed to its continuous trace.\n")
                if self.bottom_faults_defined():
                    f.write("-- Matching bottom fault traces (*.flb) are honored where\n")
                    f.write("-- defined; FAULTS records identify the simulator fault faces.\n")
                else:
                    f.write("-- No bottom fault traces (*.flb) were found; unmatched fault\n")
                    f.write("-- pieces therefore remain vertical.\n")

            f.write("\nFAULTS\n")
            f.write("-- NAME IX1 IX2 IY1 IY2 IZ1 IZ2 FACE\n")
            for name, i, j, k1, k2, face in records:
                f.write(
                    f"  '{name}' {i} {i} {j} {j} "
                    f"{k1} {k2} '{face}' /\n"
                )
            f.write("/\n")
