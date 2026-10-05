"""Unit tests for pipeline contracts, execution context, and step orchestration."""

import unittest
from pathlib import Path

from src.core.config import DocuMindConfig
from src.core.exceptions import PipelineStepError
from src.core.models import (
    Document,
    DocumentMetadata,
    DocumentPage,
    ExtractedField,
    ProcessingResult,
    Provenance,
)
from src.core.types import DocumentType, ExtractionMethod, FieldType
from src.pipeline.context import PipelineExecutionContext
from src.pipeline.runner import BasePipelineStep, PipelineRunner


class MockIngestionStep(BasePipelineStep):
    """Synthetic test step simulating document ingestion."""

    @property
    def name(self) -> str:
        return "ingestion"

    def execute(self, context: PipelineExecutionContext) -> None:
        metadata = DocumentMetadata(
            document_id="test_doc_001",
            filename=Path(context.source_file_path).name,
            file_path=context.source_file_path,
            file_type="application/pdf",
            file_size_bytes=12000,
            checksum_sha256="abc123456",
            page_count=1,
        )
        context.document = Document(
            metadata=metadata,
            pages=[DocumentPage(page_number=1, raw_text="Sample raw text for unit test.")],
            classified_type=DocumentType.INVOICE,
        )


class MockExtractionStep(BasePipelineStep):
    """Synthetic test step simulating field extraction."""

    @property
    def name(self) -> str:
        return "extraction"

    def execute(self, context: PipelineExecutionContext) -> None:
        if context.document is None:
            raise ValueError("Document not initialized.")

        prov = Provenance(
            document_id=context.document.id,
            page_number=1,
            raw_text="INV-100",
            extraction_method=ExtractionMethod.MOCK,
        )
        field = ExtractedField(
            name="invoice_number",
            value="INV-100",
            field_type=FieldType.IDENTIFIER,
            provenance=prov,
            extraction_confidence=0.95,
        )
        context.fields["invoice_number"] = field


class FailingStep(BasePipelineStep):
    """Synthetic test step that intentionally raises an exception."""

    @property
    def name(self) -> str:
        return "failing_step"

    def execute(self, context: PipelineExecutionContext) -> None:
        raise RuntimeError("Simulated internal step exception")


class TestPipelineContracts(unittest.TestCase):
    """Test suite verifying pipeline contracts and runner orchestration."""

    def test_pipeline_execution_success(self) -> None:
        runner = PipelineRunner()
        runner.add_step(MockIngestionStep())
        runner.add_step(MockExtractionStep())

        result = runner.run("data/samples/sample_invoice.pdf")

        self.assertIsInstance(result, ProcessingResult)
        self.assertEqual(result.document.id, "test_doc_001")
        self.assertIn("invoice_number", result.fields)
        self.assertEqual(result.fields["invoice_number"].value, "INV-100")
        self.assertIn("ingestion", result.execution_metadata["step_timings_seconds"])
        self.assertIn("extraction", result.execution_metadata["step_timings_seconds"])
        self.assertIsNotNone(result.summary)
        self.assertEqual(result.summary.total_fields_extracted, 1)

    def test_pipeline_step_failure_raises_pipeline_step_error(self) -> None:
        runner = PipelineRunner()
        runner.add_step(MockIngestionStep())
        runner.add_step(FailingStep())

        with self.assertRaises(PipelineStepError) as ctx:
            runner.run("data/samples/sample.pdf")

        self.assertEqual(ctx.exception.step_name, "failing_step")
        self.assertIn("Simulated internal step exception", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
