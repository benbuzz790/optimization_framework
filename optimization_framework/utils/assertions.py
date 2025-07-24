"""
NASA-style production assertions for the optimization framework.

This module will contain:
- production_assert function
- Validation utilities
- Error handling patterns

To be implemented by Achievement 2: Core Components System.
"""

# Placeholder - will be implemented by parallel development branch
def production_assert(condition, message, error_type=AssertionError):
    """Placeholder for NASA-style production assert."""
    if not condition:
        raise error_type(f"PRODUCTION ASSERT FAILED: {message}")
def validate_string(value, name, allow_empty=True):
    """
    Validate that a value is a string with optional empty check.

    Args:
        value: The value to validate
        name: Name of the parameter for error messages
        allow_empty: Whether to allow empty strings

    Raises:
        TypeError: If value is not a string
        ValueError: If value is empty and allow_empty is False
    """
    production_assert(
        isinstance(value, str),
        f"{name} must be a string, got {type(value).__name__}",
        TypeError
    )

    if not allow_empty:
        production_assert(
            len(value.strip()) > 0,
            f"{name} cannot be empty",
            ValueError
        )