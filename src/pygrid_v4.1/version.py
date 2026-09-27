"""PyGRID version/build information."""

PYGRID_VERSION = "4.1-test"
PYGRID_BUILD = "2026-09-18 experimental-v4.1-fault-band-relaxation"
FAULT_GEOMETRY_REVISION = "V4_1_HARMONIC_FAULT_BAND"


def version_text():
    return f"PyGRID {PYGRID_VERSION} | Build: {PYGRID_BUILD} | Fault geometry: {FAULT_GEOMETRY_REVISION}"


def print_banner():
    line = "=" * 72
    print(line)
    print(f" PyGRID {PYGRID_VERSION}")
    print(f" Build: {PYGRID_BUILD}")
    print(f" Fault geometry: {FAULT_GEOMETRY_REVISION}")
    print(line)
