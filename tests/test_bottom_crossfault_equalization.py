import numpy as np

from pygrid import GridModel
from pygrid.faults import FaultSet, FaultTrace
from pygrid.grdecl import GRDECLWriter


class _ConstantMap:
    def __init__(self, value):
        self.value = float(value)

    def interpolate(self, xy):
        q = np.atleast_2d(np.asarray(xy, dtype=float))
        return np.full(len(q), self.value, dtype=float)

    def interpolate_faulted(
        self,
        xy,
        faults,
        search_radius=None,
        per_line=3,
        max_neighbours=8,
        barrier_origins=None,
    ):
        q = np.atleast_2d(np.asarray(xy, dtype=float))
        return np.full(len(q), self.value, dtype=float)


def _model(tmp_path):
    (tmp_path / "case.ext").write_text(
        "0 6 0\n0 0 0\n6 0 0\n6 6 0\n", encoding="utf-8"
    )

    model = GridModel("case", "METRES", 0, 0, 6, 6, 6, 6)
    model.extent = str(tmp_path / "case.ext")
    model.layers = 2
    model.layer_mode = "PROPORTIONAL"
    model.layer_thicknesses = [1.0, 1.0]
    model.split_faults = True
    model.slanted_fault_mode = "COORD"
    model.zigzag = True
    model.slanted_fault_relax_lines = 4
    model.fault_set = FaultSet(
        traces=[
            FaultTrace(
                "F1", np.asarray([[1.0, 3.0], [5.0, 3.0]], dtype=float)
            )
        ],
        bottom_traces=[
            FaultTrace(
                "F1", np.asarray([[1.0, 4.0], [5.0, 4.0]], dtype=float)
            )
        ],
    )
    model.top_map = _ConstantMap(100.0)
    model.bottom_map = _ConstantMap(200.0)
    return model


def test_bottom_equalisation_moves_only_across_fault_not_along_strike(tmp_path):
    model = _model(tmp_path)
    writer = GRDECLWriter(model)

    base = writer._pillar_array()
    lower = writer._bottom_pillar_xy()
    delta = lower - base

    # A horizontal Y+ staircase fault is normal to J.  Therefore bottom-grid
    # equalisation may spread into other rows, but it must not spread beyond
    # the I-columns actually occupied by the fault endpoints.
    fault_keys = set(writer._fault_pillar_names())
    fault_i = {i for _j, i in fault_keys}

    moved = np.argwhere(np.linalg.norm(delta, axis=2) > 1.0e-9)
    assert len(moved) > len(fault_keys)
    assert {int(i) for _j, i in moved}.issubset(fault_i)

    # The directly faulted lower sticks hit the FLB shift exactly (+1 in Y).
    for j, i in fault_keys:
        assert np.allclose(delta[j, i], [0.0, 1.0])


def test_coord_endpoints_are_written_on_structural_top_and_bottom(tmp_path):
    model = _model(tmp_path)
    writer = GRDECLWriter(model)

    coord_top, coord_bottom = writer._coord_pillars()
    base = writer._pillar_array()
    lower = writer._bottom_pillar_xy()

    # TOP stays exactly on the original grid; BOTTOM uses the relaxed lower grid.
    assert np.allclose(coord_top[:, :, :2], base)
    assert np.allclose(coord_bottom[:, :, :2], lower)

    # COORD no longer uses artificial 0..100000 extensions for a complete grid.
    assert np.allclose(coord_top[:, :, 2], 100.0)
    assert np.allclose(coord_bottom[:, :, 2], 200.0)
