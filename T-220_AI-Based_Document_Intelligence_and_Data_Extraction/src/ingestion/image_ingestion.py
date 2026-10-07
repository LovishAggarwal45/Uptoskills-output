"""Image document ingestion handler for PNG, JPG, TIFF, and BMP inputs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from src.core.config import IngestionConfig
from src.core.exceptions import ImageDecodingError
from src.core.logging import get_logger
from src.core.models import Document, DocumentMetadata, DocumentPage
from src.core.types import DocumentType
from src.ingestion.validators import (
    compute_sha256,
    sniff_file_format,
    validate_image_decodable,
)

logger = get_logger("ingestion.image")


class ImageIngestionHandler:
    """Handles raster image validation, single-page normalization, and provenance preservation."""

    def __init__(self, config: Optional[IngestionConfig] = None) -> None:
        self.config = config or IngestionConfig()

    def ingest_image(
        self,
        file_path: Path,
        document_id: str,
        output_dir: Path,
    ) -> Document:
        """Process a single image file, copy it safely as page 1, and construct a Document.

        Args:
            file_path: Validated input image path.
            document_id: Unique identifier for the document run.
            output_dir: Destination folder where page images are staged.

        Returns:
            Document aggregate with a single DocumentPage.
        """
        mime_type, canonical_ext = sniff_file_format(file_path)
        width, height, mode, dpi = validate_image_decodable(file_path)
        sha256_hash = compute_sha256(file_path)
        file_size = file_path.stat().st_size

        # Create destination directory for page images
        doc_output_dir = output_dir / document_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        pages: List[DocumentPage] = []

        # Safely convert/save image frame(s) to standard PNG format without destroying source
        try:
            with Image.open(file_path) as img:
                n_frames = getattr(img, "n_frames", 1)
                for frame_idx in range(n_frames):
                    if n_frames > 1:
                        img.seek(frame_idx)

                    page_num = frame_idx + 1
                    target_page_path = doc_output_dir / f"page_{page_num:03d}_original.png"

                    frame_img = img.copy()
                    if frame_img.mode in ("RGBA", "RGB", "L"):
                        frame_img.save(target_page_path, format="PNG")
                    elif frame_img.mode == "CMYK":
                        frame_img.convert("RGB").save(target_page_path, format="PNG")
                    elif frame_img.mode == "P":
                        frame_img.convert("RGBA" if "transparency" in frame_img.info else "RGB").save(
                            target_page_path, format="PNG"
                        )
                    else:
                        frame_img.convert("RGB").save(target_page_path, format="PNG")

                    f_width, f_height = frame_img.size
                    f_dpi = dpi or self.config.render_dpi

                    pages.append(
                        DocumentPage(
                            page_number=page_num,
                            width=float(f_width),
                            height=float(f_height),
                            dpi=f_dpi,
                            image_path=target_page_path,
                            metadata={
                                "source_format": canonical_ext,
                                "source_color_mode": frame_img.mode,
                                "original_filename": file_path.name,
                                "frame_index": frame_idx,
                                "ingestion_status": "success",
                            },
                        )
                    )
        except Exception as e:
            raise ImageDecodingError(
                f"Failed to stage original image '{file_path.name}' to {doc_output_dir}: {e}",
                details={"file_path": str(file_path), "original_error": str(e)},
            ) from e

        logger.info(
            f"Ingested image '{file_path.name}' ({len(pages)} page(s), initial {width}x{height}, mode={mode}) -> {doc_output_dir.name}"
        )

        metadata = DocumentMetadata(
            document_id=document_id,
            filename=file_path.name,
            file_path=file_path,
            file_type=mime_type,
            file_size_bytes=file_size,
            checksum_sha256=sha256_hash,
            page_count=len(pages),
            is_scanned=True,  # Raw raster images are treated as scanned/raster inputs
        )

        return Document(
            metadata=metadata,
            pages=pages,
            classified_type=DocumentType.UNKNOWN,
        )
