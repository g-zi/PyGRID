from pathlib import Path

from pygrid.faults import FaultSet


def _write(path, text):
    Path(path).write_text(text, encoding="utf-8")


def test_tinyecl_fault_names_are_never_taken_from_bottom_record_order(tmp_path):
    flt = tmp_path / "case.flt"
    flb = tmp_path / "case.flb"

    _write(
        flt,
        """*
0 0 F1
1 0 F1
0 1 F2
1 1 F2
0 2 F3
1 2 F3
2 2 F1
3 2 F1
""",
    )
    _write(
        flb,
        """*
0 1.1 F2
1 1.1 F2
0 2.2 F3
1 2.2 F3
""",
    )

    faults = FaultSet.from_tinyecl(flt, flb)

    assert [trace.name for trace in faults.traces] == ["F1", "F2", "F3", "F1"]
    assert faults.names == ["F1", "F2", "F3"]
    assert faults.bottom_names == ["F2", "F3"]
    assert faults.slanted_names == ["F2", "F3"]
    assert all(not trace.name.startswith("FAULT_") for trace in faults.traces)
