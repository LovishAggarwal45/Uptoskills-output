"""Pipeline orchestrator for coordinating document intelligence stages."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from src.core.config import DocuMindConfig
from src.core.exceptions import PipelineExecutionError, PipelineStepError
from src.core.logging import get_logger
from src.core.models import ProcessingResult
from src.pipeline.context import PipelineExecutionContext

logger = get_logger("pipeline")


class BasePipelineStep(ABC):
    """Abstract base class for a discrete stage in the document processing pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the pipeline step."""
        pass

    @abstractmethod
    def execute(self, context: PipelineExecutionContext) -> None:
        """Execute step logic and mutate context with generated artifacts.

        Args:
            context: Shared pipeline execution context.
        """
        pass


class PipelineRunner:
    """Orchestrates execution of ordered pipeline steps against target documents."""

    def __init__(
        self,
        config: Optional[DocuMindConfig] = None,
        steps: Optional[List[BasePipelineStep]] = None,
    ) -> None:
        self.config = config or DocuMindConfig()
        self.steps: List[BasePipelineStep] = steps or []

    def add_step(self, step: BasePipelineStep) -> "PipelineRunner":
        """Append a processing step to the pipeline sequence."""
        self.steps.append(step)
        return self

    def run(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Execute full pipeline sequence on a target document.

        Args:
            file_path: Path to the input document (PDF or image).

        Returns:
            ProcessingResult containing the final document structure, fields,
            tables, validations, and review flags.
        """
        path = Path(file_path)
        logger.info(f"Initiating pipeline execution for document: {path.name}")

        context = PipelineExecutionContext(
            source_file_path=path,
            config=self.config,
        )

        for step in self.steps:
            step_name = step.name
            logger.info(f"Starting pipeline step: [{step_name}]")
            step_start = time.perf_counter()

            try:
                step.execute(context)
                elapsed = time.perf_counter() - step_start
                context.step_timings[step_name] = round(elapsed, 4)
                logger.info(f"Completed pipeline step: [{step_name}] in {elapsed:.4f}s")
            except Exception as e:
                elapsed = time.perf_counter() - step_start
                context.step_timings[step_name] = round(elapsed, 4)
                err_msg = f"Error in step '{step_name}': {str(e)}"
                logger.error(err_msg, exc_info=True)
                context.errors.append(err_msg)
                raise PipelineStepError(
                    step_name=step_name,
                    message=str(e),
                    original_exception=e,
                ) from e

        result = context.to_processing_result()
        logger.info(
            f"Pipeline execution finished for {path.name} in "
            f"{result.summary.processing_time_seconds if result.summary else 0.0:.2f}s "
            f"with {len(result.fields)} fields and {len(result.review_flags)} review flags."
        )
        return result
