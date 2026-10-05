"""Validation utilities for file integrity, MIME sniffing, image decoding, and dependencies."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.core.exceptions import (
    CorruptedDocumentError,
    EmptyFileError,
    FileSizeExceededError,
    ImageDecodingError,
    IngestionError,
    UnsupportedFileTypeError,
)

# Canonical magic byte signatures for strict binary file type sniffing
MAGIC_BYTES_SIGNATURES: Dict[str, bytes] = {
    "application/pdf": b"%PDF-",
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
    "image/tiff_le": b"II*\x00",
    "image/tiff_be": b"MM\x00*",
    "image/bmp": b"BM",
}


def validate_file_exists_and_readable(file_path: Union[str, Path]) -> Path:
    """Verify that the target path points to an existing, readable file.

    Args:
        file_path: Target path to test.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        IngestionError: If the path is a directory or not readable.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document file not found at path: {path}")
    if not path.is_file():
        raise IngestionError(f"Specified path is a directory, not a file: {path}")
    if not os.access(path, os.R_OK):
        raise IngestionError(f"Permission denied: Unable to read file at {path}")
    return path


def validate_file_not_empty(path: Path) -> int:
    """Ensure the file has non-zero byte size.

    Args:
        path: Path to the target file.

    Returns:
        File size in bytes.

    Raises:
        EmptyFileError: If file size is 0 bytes.
    """
    size = path.stat().st_size
    if size == 0:
        raise EmptyFileError(
            f"Document file is empty (0 bytes): {path.name}",
            details={"file_path": str(path)},
        )
    return size


def validate_file_size_limit(path: Path, max_bytes: int) -> int:
    """Ensure file does not exceed maximum allowable ingestion size.

    Args:
        path: Path to target file.
        max_bytes: Maximum allowed size in bytes.

    Returns:
        File size in bytes.

    Raises:
        FileSizeExceededError: If file size exceeds max_bytes.
    """
    size = path.stat().st_size
    if size > max_bytes:
        raise FileSizeExceededError(
            f"File '{path.name}' size ({size / (1024*1024):.2f} MB) exceeds maximum allowed ({max_bytes / (1024*1024):.2f} MB)",
            details={"file_path": str(path), "size_bytes": size, "max_bytes": max_bytes},
        )
    return size


def sniff_file_format(path: Path) -> Tuple[str, str]:
    """Determine MIME type and canonical extension by inspecting file header bytes.

    Do not rely solely on user-supplied file extensions.

    Args:
        path: Path to the target file.

    Returns:
        Tuple of (mime_type, canonical_extension), e.g. ("application/pdf", ".pdf").

    Raises:
        UnsupportedFileTypeError: If header bytes do not match supported document types.
    """
    with open(path, "rb") as f:
        header = f.read(16)

    if header.startswith(MAGIC_BYTES_SIGNATURES["application/pdf"]):
        return "application/pdf", ".pdf"
    elif header.startswith(MAGIC_BYTES_SIGNATURES["image/png"]):
        return "image/png", ".png"
    elif header.startswith(MAGIC_BYTES_SIGNATURES["image/jpeg"]):
        return "image/jpeg", ".jpg"
    elif header.startswith(MAGIC_BYTES_SIGNATURES["image/tiff_le"]) or header.startswith(MAGIC_BYTES_SIGNATURES["image/tiff_be"]):
        return "image/tiff", ".tiff"
    elif header.startswith(MAGIC_BYTES_SIGNATURES["image/bmp"]):
        return "image/bmp", ".bmp"

    # Fallback to extension check with caution if magic bytes are ambiguous
    ext = path.suffix.lower()
    raise UnsupportedFileTypeError(
        f"File '{path.name}' contains unsupported header bytes (detected extension: '{ext}'). "
        "DocuMind supports PDF, PNG, JPG/JPEG, TIFF, and BMP formats.",
        details={"file_path": str(path), "extension": ext, "header_hex": header[:8].hex()},
    )


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 cryptographic checksum for immutable file provenance."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def validate_image_decodable(path: Path) -> Tuple[int, int, str, Optional[int]]:
    """Verify that an image file can be decoded cleanly by Pillow.

    Args:
        path: Path to image.

    Returns:
        Tuple of (width, height, mode, dpi).

    Raises:
        ImageDecodingError: If the image cannot be decoded or has invalid dimensions.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # Verify structural integrity

        # Reopen to read metadata after verify() closes the file
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            dpi_info = img.info.get("dpi")
            dpi = int(dpi_info[0]) if dpi_info and isinstance(dpi_info, tuple) else None

            if width <= 0 or height <= 0:
                raise ImageDecodingError(
                    f"Image '{path.name}' has invalid dimensions: width={width}, height={height}",
                    details={"file_path": str(path), "width": width, "height": height},
                )
            return width, height, mode, dpi
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ImageDecodingError(
            f"Failed to decode image file '{path.name}': {str(e)}",
            details={"file_path": str(path), "original_error": str(e)},
        ) from e


def validate_pdf_structure(path: Path) -> Tuple[int, bool, Dict[str, str]]:
    """Validate that a PDF file can be parsed and inspect its structure.

    Args:
        path: Path to PDF.

    Returns:
        Tuple of (page_count, is_encrypted, metadata_dict).

    Raises:
        CorruptedDocumentError: If the PDF structure is corrupted or unreadable.
    """
    try:
        reader = PdfReader(str(path))
        is_encrypted = reader.is_encrypted
        page_count = len(reader.pages)

        if page_count == 0:
            raise CorruptedDocumentError(
                f"PDF file '{path.name}' contains 0 pages.",
                details={"file_path": str(path)},
            )

        metadata: Dict[str, str] = {}
        if reader.metadata:
            for k, v in reader.metadata.items():
                if v is not None:
                    metadata[str(k).replace("/", "")] = str(v)

        return page_count, is_encrypted, metadata
    except (PdfReadError, Exception) as e:
        if isinstance(e, CorruptedDocumentError):
            raise
        raise CorruptedDocumentError(
            f"Corrupted or unreadable PDF document '{path.name}': {str(e)}",
            details={"file_path": str(path), "original_error": str(e)},
        ) from e


def detect_poppler_path(configured_path: Optional[str] = None) -> Optional[str]:
    """Detect whether Poppler utilities (pdftoppm / pdftocairo) are available.

    Handles explicit configuration paths (directories or executable file paths),
    environment variables (POPPLER_PATH, POPPLER_DIR, POPPLER_ROOT, POPPLER_BIN),
    standard Windows release subfolder structures (e.g. Library/bin or bin),
    and system PATH lookups. Always returns the resolved directory path containing
    the binaries so pdf2image can execute pdftoppm/pdfinfo cleanly.

    Args:
        configured_path: Optional user-specified Poppler bin directory or executable.

    Returns:
        String path to Poppler directory containing binaries if found, None otherwise.
    """
    candidates_to_check: List[str] = []

    # 1. Configured path from config
    if configured_path:
        cleaned = str(configured_path).strip().strip("'\"")
        if cleaned:
            candidates_to_check.append(cleaned)

    # 2. Environment variables
    for env_var in ("POPPLER_PATH", "POPPLER_DIR", "POPPLER_ROOT", "POPPLER_BIN"):
        val = os.environ.get(env_var)
        if val:
            cleaned_env = val.strip().strip("'\"")
            if cleaned_env and cleaned_env not in candidates_to_check:
                candidates_to_check.append(cleaned_env)

    # Check candidates
    for candidate in candidates_to_check:
        try:
            p = Path(candidate)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            else:
                p = p.resolve()

            if not p.exists():
                continue

            # If user provided direct executable path (e.g. .../pdftoppm.exe or .../pdftoppm)
            if p.is_file():
                if p.stem.lower() in ("pdftoppm", "pdfinfo", "pdftocairo") or (p.parent / "pdftoppm.exe").exists() or (p.parent / "pdftoppm").exists():
                    return str(p.parent)

            # If user provided directory
            if p.is_dir():
                # Direct folder contains binary
                if (p / "pdftoppm.exe").exists() or (p / "pdftoppm").exists() or (p / "pdfinfo.exe").exists() or (p / "pdfinfo").exists():
                    return str(p)

                # Check nested Library/bin (typical Windows binary release structure)
                lib_bin = p / "Library" / "bin"
                if lib_bin.is_dir() and ((lib_bin / "pdftoppm.exe").exists() or (lib_bin / "pdftoppm").exists()):
                    return str(lib_bin)

                # Check nested bin/
                bin_dir = p / "bin"
                if bin_dir.is_dir() and ((bin_dir / "pdftoppm.exe").exists() or (bin_dir / "pdftoppm").exists()):
                    return str(bin_dir)
        except Exception:
            continue

    # 3. Check system PATH via shutil.which
    found = shutil.which("pdftoppm") or shutil.which("pdftoppm.exe")
    if found:
        try:
            return str(Path(found).resolve().parent)
        except Exception:
            return str(Path(found).parent)

    return None
