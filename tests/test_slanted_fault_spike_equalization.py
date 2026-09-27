import numpy as np

from pygrid.faults import FaultTrace
from pygrid.grdecl import GRDECLWriter


class _Model:
    equalize_slanted_fault_spikes = True
    slanted_spike_corner_fraction = 0.25
    slanted_spike_error_ratio = 1.75
    slanted_spike_relative_jump = 0.20


def _record(key, trace, station, ztop, zbottom):
    return {
        "key": key,
        "name": "F3",
        "base_xy": np.asarray([float(key[1]), float(key[0])]),
        "displacement": np.asarray([-100.0, -90.0]),
        "top_trace": trace,
        "station": station,
        "depths": (ztop, zbottom),
        "z_fault_top": float(ztop),
        "z_fault_bottom": float(zbottom),
    }


def test_only_local_staircase_outlier_is_equalized():
    trace = FaultTrace(
        name="F3",
        points=np.asarray([[0.0, 0.0], [1000.0, 0.0]], dtype=float),
    )

    # The two middle records represent two stair-step grid pillars at almost
    # the same along-fault station.  One follows the local trend, the other has
    # a large bottom-depth excursion and should be the only corrected record.
    left = _record((10, 10), trace, 0.40, 15210.0, 15860.0)
    good = _record((10, 11), trace, 0.50, 15200.0, 15840.0)
    bad = _record((11, 11), trace, 0.505, 15230.0, 15450.0)
    right = _record((11, 12), trace, 0.60, 15260.0, 15720.0)

    writer = GRDECLWriter(_Model())
    writer._grid_spacing_cache = 100.0
    writer._equalize_slanted_fault_spikes(
        {"F3": [left, good, bad, right]}
    )

    assert bad["z_fault_top"] == good["z_fault_top"]
    assert bad["z_fault_bottom"] == good["z_fault_bottom"]
    assert np.allclose(bad["displacement"], good["displacement"])
    assert bad["equalized_from"] == good["key"]

    # Good and along-fault neighbours are not altered.
    assert good["z_fault_bottom"] == 15840.0
    assert left["z_fault_bottom"] == 15860.0
    assert right["z_fault_bottom"] == 15720.0
