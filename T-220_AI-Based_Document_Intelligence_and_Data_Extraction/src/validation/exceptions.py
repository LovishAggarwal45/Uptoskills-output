"""Validation subsystem exceptions."""

from typing import Optional


class ValidationError(Exception):
    """Base exception for all validation subsystem errors."""

    def __init__(self, message: str, rule_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.rule_id = rule_id

    def __str__(self) -> str:
        if self.rule_id:
            return f"[{self.rule_id}] {self.message}"
        return self.message


class ValidationConfigurationError(ValidationError):
    """Raised when validation engine or rule configuration is invalid."""
    pass


class ValidationRuleError(ValidationError):
    """Raised when an individual validation rule fails during execution."""
    pass


class ValidationDataError(ValidationError):
    """Raised when input data passed to validation is malformed or unparseable."""
    pass


__all__ = [
    "ValidationError",
    "ValidationConfigurationError",
    "ValidationRuleError",
    "ValidationDataError",
]
