"""JSON serialization and artifact persistence for visual evidence manifests and summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.core.logging import get_logger
from src.visualization.models import VisualizationManifest, VisualizationResult

logger = get_logger("visualization.serializer")


class VisualizationSerializer:
    """Serializes and persists visual manifests, summaries, and artifact references."""

    @classmethod
    def save_manifest(
        cls,
        manifest: VisualizationManifest,
        output_dir: Union[str, Path] = "outputs/visualizations",
        filename: str = "evidence_manifest.json",
        indent: int = 2,
    ) -> Path:
        """Save the evidence manifest as a formatted JSON document.

        Args:
            manifest: VisualizationManifest instance.
            output_dir: Base visualization output directory.
            filename: Target JSON filename.
            indent: JSON indentation spaces.

        Returns:
            Path to the written JSON manifest file.
        """
        target_dir = Path(output_dir) / manifest.document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / filename

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(manifest.to_json(indent=indent))

        logger.info(f"Saved evidence manifest ({manifest.total_pages} pages, {len(manifest.evidence_regions)} spatial items) -> {out_path}")
        return out_path

    @classmethod
    def save_summary(
        cls,
        result: VisualizationResult,
        output_dir: Union[str, Path] = "outputs/visualizations",
        filename: str = "visualization_summary.json",
        indent: int = 2,
    ) -> Path:
        """Save the visualization summary and artifact mapping as a formatted JSON document.

        Args:
            result: VisualizationResult instance.
            output_dir: Base visualization output directory.
            filename: Target JSON filename.
            indent: JSON indentation spaces.

        Returns:
            Path to the written JSON summary file.
        """
        target_dir = Path(output_dir) / result.document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / filename

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.to_json(indent=indent))

        logger.info(f"Saved visualization summary ({result.total_pages} pages rendered) -> {out_path}")
        return out_path

    @classmethod
    def load_manifest(cls, file_path: Union[str, Path]) -> VisualizationManifest:
        """Load and deserialize an evidence manifest JSON file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        return VisualizationManifest.from_dict(data)

    @classmethod
    def load_result(cls, file_path: Union[str, Path]) -> VisualizationResult:
        """Load and deserialize a visualization summary JSON file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Visualization result file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        return VisualizationResult.from_dict(data)


__all__ = ["VisualizationSerializer"]
