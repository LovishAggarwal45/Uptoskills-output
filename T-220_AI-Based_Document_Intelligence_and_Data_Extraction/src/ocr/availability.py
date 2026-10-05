"""OCR engine availability detection, binary inspection, and environment diagnostics."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import OCRConfig
from src.core.logging import get_logger

logger = get_logger("ocr.availability")

# Common fallback installation paths on Windows and UNIX platforms
COMMON_TESSERACT_PATHS: List[str] = [
    # Windows standard paths
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    # UNIX / macOS standard paths
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
]


@dataclass
class OCRAvailabilityResult:
    """Detailed diagnostic assessment of the local OCR runtime environment."""
    is_wrapper_available: bool
    is_engine_available: bool
    executable_path: Optional[str] = None
    engine_version: Optional[str] = None
    available_languages: List[str] = field(default_factory=list)
    requested_language: str = "eng"
    is_language_available: bool = False
    is_ready: bool = False
    status_message: str = "Unchecked"
    diagnostic_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic result to dictionary."""
        return asdict(self)


def find_tesseract_binary(configured_path: Optional[str] = None) -> Optional[str]:
    """Locate the Tesseract OCR executable across configuration, environment, PATH, and standard directories.

    Search priority:
    1. Explicit path in application configuration
    2. TESSERACT_CMD environment variable
    3. System PATH lookup via shutil.which('tesseract')
    4. Known platform installation directories

    Args:
        configured_path: Optional explicit executable path.

    Returns:
        Absolute path to the executable if found and valid, None otherwise.
    """
    # 1. Configured path
    if configured_path:
        cp = Path(configured_path).resolve()
        if cp.is_file() and os.access(cp, os.X_OK | os.R_OK):
            return str(cp)
        elif cp.is_dir():
            exe_candidate = cp / ("tesseract.exe" if os.name == "nt" else "tesseract")
            if exe_candidate.is_file() and os.access(exe_candidate, os.X_OK | os.R_OK):
                return str(exe_candidate)

    # 2. Environment variable
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        ep = Path(env_cmd).resolve()
        if ep.is_file() and os.access(ep, os.X_OK | os.R_OK):
            return str(ep)

    # 3. System PATH
    path_which = shutil.which("tesseract")
    if path_which:
        return str(Path(path_which).resolve())

    # 4. Known platform standard paths
    for candidate in COMMON_TESSERACT_PATHS:
        try:
            cand_p = Path(candidate).resolve()
            if cand_p.is_file() and os.access(cand_p, os.X_OK | os.R_OK):
                return str(cand_p)
        except Exception:
            continue

    return None


def check_ocr_availability(config: Optional[OCRConfig] = None) -> OCRAvailabilityResult:
    """Perform a rigorous, non-destructive check of OCR wrapper, binary, version, and language data.

    Args:
        config: Optional OCRConfig instance.

    Returns:
        OCRAvailabilityResult detailing environment health.
    """
    cfg = config or OCRConfig()
    requested_lang = cfg.languages[0] if cfg.languages else "eng"

    # Step 1: Check Python wrapper (pytesseract)
    is_wrapper_ok = False
    pytesseract_mod = None
    try:
        import pytesseract
        pytesseract_mod = pytesseract
        is_wrapper_ok = True
    except ImportError:
        logger.warning("Python 'pytesseract' package is not installed.")

    # Step 2: Check Tesseract binary executable
    tesseract_bin = find_tesseract_binary(cfg.tesseract_cmd)
    if not tesseract_bin:
        return OCRAvailabilityResult(
            is_wrapper_available=is_wrapper_ok,
            is_engine_available=False,
            requested_language=requested_lang,
            is_ready=False,
            status_message=(
                "Tesseract executable was not found. "
                "On Windows, download Tesseract OCR and set 'ocr.tesseract_cmd' in configs/default.yaml "
                "or define the TESSERACT_CMD environment variable."
            ),
            diagnostic_details={
                "search_paths_checked": COMMON_TESSERACT_PATHS,
                "configured_path": cfg.tesseract_cmd,
            },
        )

    # If wrapper is available, point it to the discovered binary
    if is_wrapper_ok and pytesseract_mod:
        pytesseract_mod.pytesseract.tesseract_cmd = tesseract_bin

    # Step 3: Check version and execute smoke test
    engine_version: Optional[str] = None
    try:
        res = subprocess.run(
            [tesseract_bin, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        first_line = res.stdout.splitlines()[0] if res.stdout else ""
        engine_version = first_line.strip()
    except Exception as e:
        logger.error(f"Failed to query Tesseract version from {tesseract_bin}: {e}")
        return OCRAvailabilityResult(
            is_wrapper_available=is_wrapper_ok,
            is_engine_available=False,
            executable_path=tesseract_bin,
            requested_language=requested_lang,
            is_ready=False,
            status_message=f"Tesseract binary at {tesseract_bin} failed execution: {e}",
            diagnostic_details={"error": str(e)},
        )

    # Step 4: Query available language traineddata packs
    available_languages: List[str] = []
    try:
        if is_wrapper_ok and pytesseract_mod:
            available_languages = pytesseract_mod.get_languages(config="")
        else:
            lang_res = subprocess.run(
                [tesseract_bin, "--list-langs"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            lines = [line.strip() for line in lang_res.stdout.splitlines() if line.strip()]
            # First line is usually "List of available languages"
            available_languages = [l for l in lines if not l.lower().startswith("list of")]
    except Exception as e:
        logger.warning(f"Could not query Tesseract installed languages: {e}")

    # Step 5: Check requested language availability
    # Support compound language specifications like "eng+osd"
    req_parts = requested_lang.split("+")
    is_lang_ok = all(part in available_languages for part in req_parts) if available_languages else False

    is_ready = is_wrapper_ok and (engine_version is not None) and is_lang_ok

    status_msg = "OCR Engine is READY" if is_ready else (
        f"Language '{requested_lang}' missing from Tesseract" if not is_lang_ok else "OCR Wrapper not installed"
    )

    return OCRAvailabilityResult(
        is_wrapper_available=is_wrapper_ok,
        is_engine_available=True,
        executable_path=tesseract_bin,
        engine_version=engine_version,
        available_languages=available_languages,
        requested_language=requested_lang,
        is_language_available=is_lang_ok,
        is_ready=is_ready,
        status_message=status_msg,
        diagnostic_details={
            "pytesseract_available": is_wrapper_ok,
            "tesseract_path": tesseract_bin,
            "version_string": engine_version,
        },
    )
