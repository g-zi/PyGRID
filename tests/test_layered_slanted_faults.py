
import numpy as np

from pygrid import GridModel
from pygrid.faults import FaultSet, FaultTrace
from pygrid.grdecl import GRDECLWriter


def _model(tmp_path):
    (tmp_path / "case.ext").write_text(
        "0 10 0\n0 0 0\n10 0 0\n10 10 0\n", encoding="utf-8"
    )
    model = GridModel("case", "METRES", 0, 0, 10, 10, 5, 5)
    model.extent = str(tmp_path / "case.ext")
    model.layers = 5
    model.layer_thicknesses = [1, 1, 1, 1, 1]
    model.split_faults = True
    model.fault_set = FaultSet(
        traces=[
            FaultTrace(
                "F1",
                np.asarray([[0.0, 4.0], [10.0, 4.0]], dtype=float),
            )
        ],
        bottom_traces=[
            FaultTrace(
                "F1",
                np.asarray([[0.0, 6.0], [10.0, 6.0]], dtype=float),
            )
        ],
    )
    return model


def test_default_layered_mode_keeps_all_coord_pillars_vertical(tmp_path):
    model = _model(tmp_path)
    writer = GRDECLWriter(model)

    top, bottom = writer._coord_pillars()
    base = writer._pillar_array()

    assert model.slanted_fault_mode == "LAYERED"
    assert np.allclose(top[:, :, :2], base)
    assert np.allclose(bottom[:, :, :2], base)
    assert np.allclose(top[:, :, :2], bottom[:, :, :2])


def test_layered_fault_staircase_migrates_from_flt_to_flb(tmp_path):
    model = _model(tmp_path)
    writer = GRDECLWriter(model)

    records = writer.layered_fault_faces()
    assert records

    faces_by_k = {k: set() for k in range(1, model.layers + 1)}
    for name, i, j, k1, k2, face in records:
        for k in range(k1, k2 + 1):
            faces_by_k[k].add((name, i, j, face))

    top_faces = set(writer._faces_for_fault_set(writer._fault_set_for_level("top")))
    bottom_faces = set(
        writer._faces_for_fault_set(writer._fault_set_for_level("bottom"))
    )

    assert faces_by_k[1] == top_faces
    assert faces_by_k[model.layers] == bottom_faces
    assert top_faces != bottom_faces
