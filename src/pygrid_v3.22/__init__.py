"""
PyGRID
Eclipse GRDECL grid generator
"""

from .version import (
    PYGRID_VERSION,
    PYGRID_BUILD,
    FAULT_GEOMETRY_REVISION,
    print_banner,
    version_text,
)

__version__ = PYGRID_VERSION

from .grid import GridModel
from .license import (
    PyGRIDLicenseError,
    check_tinyecl_license,
    get_machine_id,
    license_status_text,
)

# Print once when PyGRID is imported by RUNFILE.PY.  This makes it obvious
# which executable/source build is actually running.
print_banner()

__all__ = [
    "GridModel",
    "PyGRIDLicenseError",
    "check_tinyecl_license",
    "get_machine_id",
    "license_status_text",
    "PYGRID_VERSION",
    "PYGRID_BUILD",
    "FAULT_GEOMETRY_REVISION",
    "version_text",
]
