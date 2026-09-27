import numpy as np

from pygrid.maps import ContourMap
from pygrid.faults import FaultSet


def test_contour_densification_uses_contiguous_runs_not_reused_counter(tmp_path):
    path = tmp_path / "map.cnt"
    path.write_text(
        "0 0 100 7\n"
        "10 0 100 7\n"
        # Same TinyECL line counter reused by another contour level far away.
        "1000 1 200 7\n"
        "1010 1 200 7\n",
        encoding="utf-8",
    )

    contour = ContourMap(path).read().densify(2.0)

    assert contour.source_point_count == 4
    assert len(contour._line_groups) == 2
    assert len(contour.points) == 12

    xs = np.asarray([p.x for p in contour.points])
    # No artificial point may be inserted across the 990-unit gap between the
    # two independent contour lines that happen to reuse source id 7.
    assert not np.any((xs > 10.0) & (xs < 1000.0))


def test_two_top_contour_crossings_make_vertical_fault_surface_transparent(tmp_path):
    cnt = tmp_path / "top.cnt"
    flt = tmp_path / "case.flt"

    cnt.write_text(
        # Two distinct contour levels cross FZERO at x=4.
        "0 3 100 A\n"
        "6 3 100 A\n"
        "0 7 200 B\n"
        "6 7 200 B\n"
        # Only one contour crosses FTHROW at x=8.
        "6.5 5 150 C\n"
        "10 5 150 C\n",
        encoding="utf-8",
    )
    flt.write_text(
        "4 0 FZERO\n"
        "4 10 FZERO\n"
        "8 0 FTHROW\n"
        "8 10 FTHROW\n",
        encoding="utf-8",
    )

    contour = ContourMap(cnt).read().densify(1.0)
    faults = FaultSet.from_tinyecl(flt)
    faults.classify_surface_barriers(
        contour,
        min_crossings=2,
        min_station_separation=0.15,
    )

    zero = faults.named_traces("FZERO")[0]
    throwing = faults.named_traces("FTHROW")[0]

    assert zero.surface_barrier is False
    assert [z for z, _station in zero.surface_crossings] == [100.0, 200.0]
    assert throwing.surface_barrier is True

    # Zero-throw trace no longer blocks structural interpolation.
    assert faults.blocks_segment(np.array([3.0, 5.0]), np.array([5.0, 5.0])) is False
    # The genuine barrier trace still does.
    assert faults.blocks_segment(np.array([7.0, 5.0]), np.array([9.0, 5.0])) is True
