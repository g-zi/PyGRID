import numpy as np

from pygrid import GridModel
from pygrid.faults import FaultSet, FaultTrace
from pygrid.grdecl import GRDECLWriter


class _SpyMap:
    def __init__(self, value):
        self.value = float(value)
        self.calls = []

    def interpolate(self, xy):
        query = np.atleast_2d(np.asarray(xy, dtype=float))
        return np.full(len(query), self.value, dtype=float)

    def interpolate_faulted(
        self,
        xy,
        faults,
        search_radius=None,
        per_line=3,
        max_neighbours=8,
        barrier_origins=None,
    ):
        query = np.atleast_2d(np.asarray(xy, dtype=float))
        trace_y = [float(np.mean(trace.points[:, 1])) for trace in faults.traces]
        self.calls.append((query.copy(), trace_y))
        return np.full(len(query), self.value, dtype=float)


def _xy_on_coord(coord_top, coord_bottom, z):
    t = (z - coord_top[:, :, 2]) / (coord_bottom[:, :, 2] - coord_top[:, :, 2])
    return coord_top[:, :, :2] + t[:, :, None] * (
        coord_bottom[:, :, :2] - coord_top[:, :, :2]
    )


def test_top_bottom_are_built_independently_then_internal_layer_is_filled(tmp_path):
    (tmp_path / "case.ext").write_text(
        "0 10 0\n0 0 0\n10 0 0\n10 10 0\n", encoding="utf-8"
    )

    top_trace = FaultTrace(
        "F1", np.asarray([[0.0, 5.0], [10.0, 5.0]], dtype=float)
    )
    bottom_trace = FaultTrace(
        "F1", np.asarray([[0.0, 7.0], [10.0, 7.0]], dtype=float)
    )

    model = GridModel("case", "METRES", 0, 0, 10, 10, 2, 2)
    model.extent = str(tmp_path / "case.ext")
    model.layers = 2
    model.layer_mode = "PROPORTIONAL"
    model.layer_thicknesses = [1.0, 1.0]
    model.split_faults = True
    model.slanted_fault_mode = "COORD"
    model.zigzag = False
    model.fault_set = FaultSet(
        traces=[top_trace], bottom_traces=[bottom_trace]
    )
    model.top_map = _SpyMap(100.0)
    model.bottom_map = _SpyMap(200.0)

    writer = GRDECLWriter(model)
    interfaces = writer._cell_corner_interfaces()

    # Endpoint horizons are fixed first; only then is K=1 filled halfway.
    assert np.allclose(interfaces[0], 100.0)
    assert np.allclose(interfaces[1], 150.0)
    assert np.allclose(interfaces[2], 200.0)

    # TOP interpolation sees FLT, BOTTOM interpolation sees FLB.
    assert model.top_map.calls
    assert model.bottom_map.calls
    assert model.top_map.calls[0][1] == [5.0]
    assert model.bottom_map.calls[0][1] == [7.0]

    top_queries = model.top_map.calls[0][0]
    bottom_queries = model.bottom_map.calls[0][0]
    assert not np.allclose(top_queries, bottom_queries)

    # COORD is subsequently fitted through those exact endpoint grids.
    coord_top, coord_bottom = writer._coord_pillars()
    base = writer._pillar_array()
    lower = writer._bottom_pillar_xy()
    assert np.allclose(_xy_on_coord(coord_top, coord_bottom, 100.0), base)
    assert np.allclose(_xy_on_coord(coord_top, coord_bottom, 200.0), lower)
