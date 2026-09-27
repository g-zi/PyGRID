import numpy as np

from pygrid import GridModel
from pygrid.grdecl import GRDECLWriter


def test_same_name_bottom_trace_tilts_only_matching_fault_pillars(tmp_path):
    # Simple rectangular model with one slanted and one vertical fault.
    (tmp_path / "case.ext").write_text(
        "0 10 0\n0 0 0\n10 0 0\n10 10 0\n", encoding="utf-8"
    )
    (tmp_path / "case.flt").write_text(
        "0 5 F1\n10 5 F1\n2 8 F2\n8 8 F2\n", encoding="utf-8"
    )
    # Only F1 has a bottom trace; it is displaced by +1 in Y.
    (tmp_path / "case.flb").write_text(
        "0 6 F1\n10 6 F1\n", encoding="utf-8"
    )
    # Three-point maps are enough for interpolation.
    (tmp_path / "top.cnt").write_text(
        "0 0 100 A\n10 0 100 A\n0 10 100 A\n10 10 100 A\n",
        encoding="utf-8",
    )
    (tmp_path / "bottom.cnb").write_text(
        "0 0 200 A\n10 0 200 A\n0 10 200 A\n10 10 200 A\n",
        encoding="utf-8",
    )

    model = GridModel("case", "METRES", 0, 0, 10, 10, 5, 5)
    model.extent = str(tmp_path / "case.ext")
    model.top = str(tmp_path / "top.cnt")
    model.bottom = str(tmp_path / "bottom.cnb")
    model.top_faults = str(tmp_path / "case.flt")
    model.bottom_faults = str(tmp_path / "case.flb")
    model.layers = 2
    model.split_faults = True
    model.slanted_fault_mode = "COORD"
    model.read_maps()
    model.read_faults()

    writer = GRDECLWriter(model)
    coord_top, coord_bottom = writer._coord_pillars()
    delta = np.linalg.norm(coord_bottom[:, :, :2] - coord_top[:, :, :2], axis=2)

    assert model.fault_set.slanted_names == ["F1"]
    assert np.count_nonzero(delta > 1.0e-8) > 0

    # Output fault names remain the source names, including vertical F2.
    names = {row[0] for row in writer.fault_faces()}
    assert names == {"F1", "F2"}
