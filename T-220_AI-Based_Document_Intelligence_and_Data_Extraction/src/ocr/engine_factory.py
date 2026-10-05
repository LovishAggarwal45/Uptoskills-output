"""Factory for instantiating pluggable OCR engine backends."""

from __future__ import annotations

from typing import Dict, Optional, Type

from src.core.config import OCRConfig
from src.ocr.base import BaseOCREngine
from src.ocr.exceptions import OCRConfigurationError
from src.ocr.tesseract_engine import TesseractOCREngine


class OCREngineFactory:
    """Factory registry for creating configured OCR engine instances."""

    _registry: Dict[str, Type[BaseOCREngine]] = {
        "tesseract": TesseractOCREngine,
    }

    @classmethod
    def register_engine(cls, name: str, engine_cls: Type[BaseOCREngine]) -> None:
        """Register a new OCR engine backend into the factory."""
        cls._registry[name.lower()] = engine_cls

    @classmethod
    def create_engine(
        cls,
        engine_type: Optional[str] = None,
        config: Optional[OCRConfig] = None,
    ) -> BaseOCREngine:
        """Create an OCR engine instance based on engine type or configuration.

        Args:
            engine_type: Optional engine name override (e.g. 'tesseract').
            config: Optional OCRConfig instance.

        Returns:
            Configured BaseOCREngine instance.

        Raises:
            OCRConfigurationError: If the requested engine type is not registered.
        """
        cfg = config or OCRConfig()
        selected_type = (engine_type or cfg.engine_type).lower()

        engine_cls = cls._registry.get(selected_type)
        if not engine_cls:
            supported = list(cls._registry.keys())
            raise OCRConfigurationError(
                f"Unsupported OCR engine type: '{selected_type}'. Supported engines: {supported}"
            )

        return engine_cls(config=cfg)
