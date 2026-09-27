import numpy as np

from pygrid.faults import FaultSet


def test_unmatched_repeated_same_name_top_piece_stays_vertical(tmp_path):
    flt = tmp_path / "case.flt"
    flb = tmp_path / "case.flb"

    # F1 has two separated top pieces but only the first has a bottom trace.
    flt.write_text(
        "0 0 F1\n10 0 F1\n100 0 F1\n110 0 F1\n",
        encoding="utf-8",
    )
    flb.write_text(
        "0 2 F1\n10 2 F1\n",
        encoding="utf-8",
    )

    faults = FaultSet.from_tinyecl(flt, flb)

    first = faults.slant_displacement(
        "F1", np.array([5.0, 0.0]), max_top_distance=10.0
    )
    second = faults.slant_displacement(
        "F1", np.array([105.0, 0.0]), max_top_distance=10.0
    )

    assert np.linalg.norm(first) > 1.0
    assert np.allclose(second, [0.0, 0.0])


def test_short_bottom_trace_does_not_collapse_long_top_tail_to_endpoint(tmp_path):
    flt = tmp_path / "case.flt"
    flb = tmp_path / "case.flb"

    # The bottom interpretation covers only the first 60% of the top fault.
    # Old nearest-endpoint logic connected the remaining top tail to (60, 2),
    # creating progressively longer, almost along-strike COORD sticks.
    flt.write_text(
        "0 0 F1\n50 0 F1\n100 0 F1\n",
        encoding="utf-8",
    )
    flb.write_text(
        "0 2 F1\n30 2 F1\n60 2 F1\n",
        encoding="utf-8",
    )

    faults = FaultSet.from_tinyecl(flt, flb)

    supported = faults.slant_displacement(
        "F1",
        np.array([30.0, 0.0]),
        max_top_distance=10.0,
        fade_distance=10.0,
    )
    unsupported = faults.slant_displacement(
        "F1",
        np.array([90.0, 0.0]),
        max_top_distance=10.0,
        fade_distance=10.0,
    )

    assert np.allclose(supported, [0.0, 2.0], atol=1.0e-8)
    assert np.allclose(unsupported, [0.0, 0.0], atol=1.0e-8)


def test_top_down_mapping_follows_top_station_through_bottom_bend(tmp_path):
    flt = tmp_path / "case.flt"
    flb = tmp_path / "case.flb"

    # FLT is the structural-top reference.  FLB has a local bend normal to
    # strike.  A nearest-point projection from x=15 on FLT would jump to the
    # vertical FLB segment near x=10; top-down station mapping must instead
    # remain aligned with the same along-fault position.
    flt.write_text(
        "0 0 F1\n10 0 F1\n20 0 F1\n",
        encoding="utf-8",
    )
    flb.write_text(
        "0 2 F1\n10 2 F1\n10 8 F1\n20 8 F1\n",
        encoding="utf-8",
    )

    faults = FaultSet.from_tinyecl(flt, flb)
    displacement = faults.slant_displacement(
        "F1",
        np.array([15.0, 0.0]),
        max_top_distance=5.0,
        fade_distance=2.0,
    )

    # The stick stays at the x=15 top station and moves downward/across fault.
    assert abs(displacement[0]) < 1.0e-8
    assert displacement[1] > 5.0
