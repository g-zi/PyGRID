"""PyGRID version/build information."""

PYGRID_VERSION = "0.3.0-dev"
PYGRID_BUILD = "2026-09-16 conforming-fault-v3.22-flt-trace-conforming"
FAULT_GEOMETRY_REVISION = "CONFORM_V3_22_FLT_TRACE_CONFORMING"


def version_text():
    return f"PyGRID {PYGRID_VERSION} | Build: {PYGRID_BUILD} | Fault geometry: {FAULT_GEOMETRY_REVISION}"


def print_banner():
    line = "=" * 72
    print(line)
    print(f" PyGRID {PYGRID_VERSION}")
    print(f" Build: {PYGRID_BUILD}")
    print(f" Fault geometry: {FAULT_GEOMETRY_REVISION}")
    print(line)
