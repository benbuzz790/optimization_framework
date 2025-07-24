"""
Binary Variable class for optimization framework.

This module provides the BinaryVariable class for binary (0/1) optimization
variables with full NASA-style validation.
"""

from typing import Any
from .variable import Variable


class BinaryVariable(Variable):
    """
    Binary (0/1) variable.

    Accepts only 0 or 1 values. Bounds are fixed to (0, 1).
    Commonly used for on/off decisions, feature selection, etc.

    Examples:
        # Binary decision variable
        use_feature = BinaryVariable("use_feature")

        # Binary selection variable
        selected = BinaryVariable("selected")
    """

    def __init__(self, name: str):
        """
        Initialize binary variable with fixed bounds (0, 1).

        Args:
            name: Variable name (must be non-empty string)
        """
        super().__init__(name, bounds=(0, 1))

    def validate_value(self, value: Any) -> bool:
        """
        Validate binary value.

        Args:
            value: Value to validate

        Returns:
            bool: True if value is 0 or 1

        NASA Assert: value is 0 or 1
        """
        return value in [0, 1]

    def get_type_name(self) -> str:
        """Return variable type identifier."""
        return "binary"