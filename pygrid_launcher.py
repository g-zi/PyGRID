"""
PyGRID Windows launcher.

Usage:
    PyGRID.exe RUNFILE.PY
"""

import os
import runpy
import sys
from pathlib import Path

import pygrid


def main():
    if len(sys.argv) < 2:
        print()
        print("PyGRID - Eclipse corner-point grid generator")
        print()
        print("Usage:")
        print("    PyGRID.exe RUNFILE.PY")
        print()
        input("Press Enter to close...")
        return 1

    runfile = Path(sys.argv[1]).resolve()

    if not runfile.exists():
        print(f"ERROR: RUNFILE not found:")
        print(f"       {runfile}")
        input("Press Enter to close...")
        return 1

    if not runfile.is_file():
        print(f"ERROR: Not a file:")
        print(f"       {runfile}")
        input("Press Enter to close...")
        return 1

    print()
    print("PyGRID")
    print("------")
    print(f"RUNFILE : {runfile}")
    print()

    # RUNFILE.PY uses relative filenames such as TinyECL.cnt,
    # TinyECL.cnb, TinyECL.ext, etc.
    # Therefore execute it from its own directory.
    old_cwd = Path.cwd()

    try:
        os.chdir(runfile.parent)

        runpy.run_path(
            str(runfile),
            run_name="__main__"
        )

    except Exception as exc:
        print()
        print("PyGRID ERROR")
        print("------------")
        print(str(exc))
        print()

        input("Press Enter to close...")
        return 1

    finally:
        os.chdir(old_cwd)

    return 0


if __name__ == "__main__":
    sys.exit(main())