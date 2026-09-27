"""TinyECL/PyGRID shared license verification.

PyGRID deliberately consumes the same ``TinyECL.lic`` file as TinyECL.
A verified TinyECL license is valid for PyGRID on the same machine.

License lookup order is intentionally forgiving: every candidate is tried and
an invalid local copy does not mask a valid installed copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Dict, Iterable, Optional


LICENSE_FILE_NAME = "TinyECL.lic"
LICENSE_VERSION = "1"
LICENSE_PRODUCT = "TinyECL"


class PyGRIDLicenseError(RuntimeError):
    """Raised when PyGRID output is requested without a usable TinyECL license."""


@dataclass(frozen=True)
class LicenseStatus:
    valid: bool
    state: str
    message: str
    path: Optional[Path] = None
    licensee: str = ""
    license_type: str = ""
    expires: str = ""
    machine_id: str = ""


_cached_status: Optional[LicenseStatus] = None


def _normalize_machine_id(value: str) -> str:
    return str(value or "").strip().upper().replace("{", "").replace("}", "")


def _valid_uuid(value: str) -> bool:
    value = _normalize_machine_id(value)
    return bool(value) and value not in {
        "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF",
        "00000000-0000-0000-0000-000000000000",
    }


def _run_quiet(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if os.name == "nt"
                else 0
            ),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def get_machine_id() -> str:
    """Return the same Windows machine identity used by TinyECL VBA.

    TinyECL queries ``Win32_ComputerSystemProduct.UUID`` through WMI.  On
    modern Windows PyGRID asks PowerShell/CIM for the same WMI class; on older
    Windows it falls back to WMIC.  The final fallback mirrors TinyECL and uses
    COMPUTERNAME.
    """
    if os.name == "nt":
        output = _run_quiet(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance -ClassName Win32_ComputerSystemProduct "
                "-ErrorAction Stop).UUID",
            ]
        )
        for line in output.splitlines():
            candidate = _normalize_machine_id(line)
            if _valid_uuid(candidate):
                return candidate

        output = _run_quiet(["wmic.exe", "csproduct", "get", "UUID", "/value"])
        for line in output.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key.strip().upper() == "UUID":
                    candidate = _normalize_machine_id(value)
                    if _valid_uuid(candidate):
                        return candidate
            else:
                candidate = _normalize_machine_id(line)
                if _valid_uuid(candidate) and candidate != "UUID":
                    return candidate

        return _normalize_machine_id(os.environ.get("COMPUTERNAME", ""))

    # PyGRID's first distribution is Windows-oriented.  This non-Windows
    # fallback is useful for source-level diagnostics/tests only.
    return _normalize_machine_id(os.environ.get("COMPUTERNAME") or platform.node())


def _appdata_license_path() -> Optional[Path]:
    """Return TinyECL's current installed-license location.

    TinyECL now installs its shared license below LOCALAPPDATA, not the older
    Roaming APPDATA location.  Keep one canonical installed-license path so
    TinyECL and PyGRID always use the same file.
    """
    localappdata = os.environ.get("LOCALAPPDATA", "").strip()
    if not localappdata:
        return None
    return Path(localappdata) / "TinyECL" / LICENSE_FILE_NAME


def _executable_folder() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def license_candidates() -> list[Path]:
    """Return unique license candidates PyGRID should try.

    TinyECL normally launches PyGRID from a project ``Grid`` subfolder while
    ``TinyECL.lic`` may sit beside the TinyECL workbook one directory above.
    Search the working folder and a few parents before falling back to the
    executable folder and the installed LocalAppData copy.
    """
    candidates: list[Path] = []

    work = Path.cwd()

    # Current Grid/project folder first.
    candidates.append(work / LICENSE_FILE_NAME)

    # Then walk upward through the project hierarchy.  Three parent levels are
    # enough for TinyECL\Grid while avoiding an unnecessarily broad disk search.
    parent = work.parent
    for _ in range(3):
        if parent == parent.parent:
            break
        candidates.append(parent / LICENSE_FILE_NAME)
        parent = parent.parent

    # A license beside the PyGRID executable/source is also accepted.
    candidates.append(_executable_folder() / LICENSE_FILE_NAME)

    # TinyECL installs a verified shared copy in LOCALAPPDATA here.
    central = _appdata_license_path()
    if central is not None:
        candidates.append(central)

    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve()).lower()
        except OSError:
            key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def _read_key_value_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", errors="strict") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("--"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip().upper()] = value.strip()
    return values


def _canonical_license_text(values: Dict[str, str]) -> str:
    return (
        f"V={values['VERSION'].strip()}|"
        f"P={values['PRODUCT'].strip().upper()}|"
        f"N={values['LICENSEE'].strip()}|"
        f"T={values['TYPE'].strip().upper()}|"
        f"M={_normalize_machine_id(values['MACHINE'])}|"
        f"E={values['EXPIRES'].strip().upper()}"
    )


def _signing_salt() -> str:
    # Kept split, matching the distributed TinyECL v1.5 verifier.
    return "".join((
        "T7p-", "4Qm9-", "x2Va-", "N8cR-",
        "6fK1-", "zP3w-", "H5sD-", "9uJ2",
    ))


def _signature_for(values: Dict[str, str]) -> str:
    payload = _canonical_license_text(values) + "|" + _signing_salt()
    return hashlib.sha256(payload.encode("utf-16le")).hexdigest().upper()


def _verify_license_file(path: Path, machine_id: str) -> LicenseStatus:
    try:
        values = _read_key_value_file(path)
    except (OSError, UnicodeError) as exc:
        return LicenseStatus(
            False,
            "INVALID",
            f"TinyECL license could not be read: {exc}",
            path=path,
            machine_id=machine_id,
        )

    required = {"VERSION", "PRODUCT", "LICENSEE", "TYPE", "MACHINE", "EXPIRES", "SIGNATURE"}
    missing = sorted(required.difference(values))
    if missing:
        return LicenseStatus(
            False,
            "INVALID",
            "TinyECL license is incomplete (missing " + ", ".join(missing) + ").",
            path=path,
            machine_id=machine_id,
        )

    if values["VERSION"].strip() != LICENSE_VERSION:
        return LicenseStatus(False, "INVALID", "Unsupported TinyECL license version.", path=path, machine_id=machine_id)

    if values["PRODUCT"].strip().upper() != LICENSE_PRODUCT.upper():
        return LicenseStatus(False, "INVALID", "This license is not for TinyECL/PyGRID.", path=path, machine_id=machine_id)

    actual = values["SIGNATURE"].strip().upper()
    expected = _signature_for(values)
    if len(expected) != 64 or actual != expected:
        return LicenseStatus(False, "INVALID", "TinyECL license signature is invalid.", path=path, machine_id=machine_id)

    licensed_machine = _normalize_machine_id(values["MACHINE"])
    if licensed_machine != "ANY" and licensed_machine != _normalize_machine_id(machine_id):
        return LicenseStatus(
            False,
            "WRONG_MACHINE",
            "This TinyECL license belongs to another computer.",
            path=path,
            machine_id=machine_id,
        )

    expires = values["EXPIRES"].strip()
    if expires.upper() != "NEVER":
        try:
            expiry_date = date.fromisoformat(expires)
        except ValueError:
            return LicenseStatus(False, "INVALID", "TinyECL license expiry date is invalid.", path=path, machine_id=machine_id)
        if date.today() > expiry_date:
            return LicenseStatus(
                False,
                "EXPIRED",
                f"TinyECL license expired on {expiry_date.isoformat()}.",
                path=path,
                licensee=values["LICENSEE"].strip(),
                license_type=values["TYPE"].strip().upper(),
                expires=expires,
                machine_id=machine_id,
            )

    license_type = values["TYPE"].strip().upper()
    if license_type == "DEVELOPER":
        state = "DEVELOPER"
    elif license_type in {"EVALUATION", "TRIAL"}:
        state = "EVALUATION"
    else:
        state = "LICENSED"

    licensee = values["LICENSEE"].strip()
    return LicenseStatus(
        True,
        state,
        f"Licensed to {licensee} ({license_type}).",
        path=path,
        licensee=licensee,
        license_type=license_type,
        expires=expires,
        machine_id=machine_id,
    )


def _install_verified_license(source: Path) -> None:
    target = _appdata_license_path()
    if target is None:
        return
    try:
        if source.resolve() == target.resolve():
            return
    except OSError:
        pass
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    except OSError:
        # Match TinyECL: failure to install the LocalAppData copy must not invalidate
        # a license that has already verified successfully.
        pass


def check_tinyecl_license(*, refresh: bool = False) -> LicenseStatus:
    """Check the shared TinyECL license and return a detailed status."""
    global _cached_status
    if _cached_status is not None and not refresh:
        return _cached_status

    machine_id = get_machine_id()
    if not machine_id:
        _cached_status = LicenseStatus(
            False,
            "INVALID",
            "PyGRID could not determine the Windows machine ID.",
        )
        return _cached_status

    attempted: list[LicenseStatus] = []
    for path in license_candidates():
        if not path.is_file():
            continue
        status = _verify_license_file(path, machine_id)
        attempted.append(status)
        if status.valid:
            _install_verified_license(path)
            _cached_status = status
            return status

    if attempted:
        # Prefer a meaningful machine/expiry error over a generic invalid file.
        priority = {"WRONG_MACHINE": 0, "EXPIRED": 1, "INVALID": 2}
        attempted.sort(key=lambda item: priority.get(item.state, 99))
        _cached_status = attempted[0]
        return _cached_status

    central = _appdata_license_path()
    location = str(central) if central is not None else "%LOCALAPPDATA%\\TinyECL\\TinyECL.lic"
    tried = "\n".join(f"  - {path}" for path in license_candidates())
    _cached_status = LicenseStatus(
        False,
        "MISSING",
        "No TinyECL license was found. PyGRID uses the same TinyECL.lic file "
        f"as TinyECL. A verified TinyECL license is normally installed at {location}."
        f"\nLicense locations tried:\n{tried}",
        machine_id=machine_id,
    )
    return _cached_status


def require_tinyecl_license(action: str = "generate PyGRID output") -> LicenseStatus:
    """Require a valid shared TinyECL license for a protected PyGRID action."""
    status = check_tinyecl_license()
    if status.valid:
        return status

    message = (
        f"PyGRID license check failed while trying to {action}.\n"
        f"{status.message}\n"
        f"Machine ID: {status.machine_id or '<unknown>'}\n"
        "Use the same TinyECL.lic file as TinyECL."
    )
    raise PyGRIDLicenseError(message)


def license_status_text(*, refresh: bool = False) -> str:
    """Human-readable license status for diagnostics."""
    status = check_tinyecl_license(refresh=refresh)
    lines = [status.message]
    if status.machine_id:
        lines.append(f"Machine ID: {status.machine_id}")
    if status.path:
        lines.append(f"License: {status.path}")
    return "\n".join(lines)
