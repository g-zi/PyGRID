import numpy as np

from pygrid import GridModel
from pygrid.grdecl import GRDECLWriter


def _xy_on_coord(coord_top, coord_bottom, z):
    t = (z - coord_top[:, :, 2]) / (coord_bottom[:, :, 2] - coord_top[:, :, 2])
    return coord_top[:, :, :2] + t[:, :, None] * (
        coord_bottom[:, :, :2] - coord_top[:, :, :2]
    )


def test_zigzag_keeps_top_regular_and_relaxes_lower_fault_shift(tmp_path):
    (tmp_path / "case.ext").write_text(
        "0 20 0\n0 0 0\n20 0 0\n20 20 0\n", encoding="utf-8"
    )
    # Horizontal top fault at y=10; lower interpretation is shifted +4 m.
    (tmp_path / "case.flt").write_text(
        "0 10 F1\n20 10 F1\n2 17 F2\n18 17 F2\n", encoding="utf-8"
    )
    # Only F1 is slanted.  F2 has no lower point and must remain straight.
    (tmp_path / "case.flb").write_text(
        "0 14 F1\n20 14 F1\n", encoding="utf-8"
    )
    (tmp_path / "top.cnt").write_text(
        "0 0 100 A\n20 0 100 A\n0 20 100 A\n20 20 100 A\n",
        encoding="utf-8",
    )
    (tmp_path / "bottom.cnb").write_text(
        "0 0 200 A\n20 0 200 A\n0 20 200 A\n20 20 200 A\n",
        encoding="utf-8",
    )

    model = GridModel("case", "METRES", 0, 0, 20, 20, 10, 10)
    model.extent = str(tmp_path / "case.ext")
    model.top = str(tmp_path / "top.cnt")
    model.bottom = str(tmp_path / "bottom.cnb")
    model.top_faults = str(tmp_path / "case.flt")
    model.bottom_faults = str(tmp_path / "case.flb")
    model.layers = 2
    model.split_faults = True
    model.slanted_fault_mode = "COORD"
    model.zigzag = True
    model.slanted_fault_relax_lines = 2
    model.slanted_fault_max_step_fraction = 1.0
    model.read_maps()
    model.read_faults()

    writer = GRDECLWriter(model)
    coord_top, coord_bottom = writer._coord_pillars()
    base = writer._pillar_array()

    top_xy = _xy_on_coord(coord_top, coord_bottom, 100.0)
    bottom_xy = _xy_on_coord(coord_top, coord_bottom, 200.0)
    shift = bottom_xy - base
    shift_size = np.linalg.norm(shift, axis=2)

    # First structural layer remains the original equal-size ZigZag grid.
    assert np.allclose(top_xy, base, atol=1.0e-8)

    # A matched F1 staircase pillar carries essentially the full +4 m shift.
    assert np.max(shift_size) > 3.9

    # The movement is not concentrated in one fault row: neighbouring rows are
    # moved by smaller amounts, so lower cell widths change progressively.
    row_max = np.max(shift_size, axis=1)
    full_rows = np.flatnonzero(row_max > 3.9)
    assert len(full_rows) >= 1
    r = int(full_rows[len(full_rows) // 2])
    neighbours = [idx for idx in (r - 1, r + 1) if 0 <= idx < len(row_max)]
    assert any(0.0 < row_max[idx] < row_max[r] for idx in neighbours)

    # Far enough away from F1 the regular lower grid is untouched.
    assert np.min(row_max[[0, -1]]) < 1.0e-8

    # F2 is not in FLB, so it is not a source of slanted displacement.
    assert model.fault_set.slanted_names == ["F1"]
