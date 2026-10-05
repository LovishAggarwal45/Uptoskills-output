"""DocuMind AI Production-Quality Table Intelligence & Line-Item Extraction Subsystem."""

from src.tables.base import (
    BaseLineItemExtractor,
    BaseTableConfidenceScorer,
    BaseTableContinuationDetector,
    BaseTableDetector,
    BaseTableExtractor,
    BaseTableReconstructor,
    BaseTableValidator,
)
from src.tables.cell_parser import CellParser
from src.tables.column_parser import ColumnParser
from src.tables.confidence import TableConfidenceScorer
from src.tables.continuation import TableContinuationDetector
from src.tables.detector import RuleBasedTableDetector
from src.tables.exceptions import (
    InvalidTableInput,
    TableConfigurationError,
    TableExtractionError,
    TableStructureError,
    TableValidationError,
)
from src.tables.extractor import TableExtractor
from src.tables.header_detector import (
    CANONICAL_HEADER_SYNONYMS,
    HeaderDetector,
    match_canonical_header,
)
from src.tables.line_item_extractor import LineItemExtractor
from src.tables.models import (
    LineItem,
    Table,
    TableCell,
    TableColumn,
    TableExtractionResult,
    TableHeader,
    TableRegion,
    TableRow,
    TableValidationResult,
    TableValueType,
)
from src.tables.row_parser import RowParser
from src.tables.structure import TableStructureReconstructor
from src.tables.validators import TableValidator

__all__ = [
    # Base interfaces
    "BaseTableDetector",
    "BaseTableReconstructor",
    "BaseLineItemExtractor",
    "BaseTableValidator",
    "BaseTableContinuationDetector",
    "BaseTableConfidenceScorer",
    "BaseTableExtractor",
    # Models
    "TableValueType",
    "TableCell",
    "TableColumn",
    "TableHeader",
    "TableRow",
    "TableRegion",
    "LineItem",
    "TableValidationResult",
    "Table",
    "TableExtractionResult",
    # Core components
    "HeaderDetector",
    "ColumnParser",
    "RowParser",
    "CellParser",
    "TableStructureReconstructor",
    "RuleBasedTableDetector",
    "LineItemExtractor",
    "TableValidator",
    "TableContinuationDetector",
    "TableConfidenceScorer",
    "TableExtractor",
    "match_canonical_header",
    "CANONICAL_HEADER_SYNONYMS",
    # Exceptions
    "TableExtractionError",
    "InvalidTableInput",
    "TableStructureError",
    "TableConfigurationError",
    "TableValidationError",
]
