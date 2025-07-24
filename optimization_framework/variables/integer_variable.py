"""
Integer Variable class for optimization framework.

This module provides the IntegerVariable class for integer-valued optimization
variables with full NASA-style validation and bounds handling.
"""

from typing import Tuple, Any
from .variable import Variable
from ..utils.assertions import production_assert


class IntegerVariable(Variable):
    """
    Integer-valued variable.

    Accepts only integer values within specified bounds.
    Default bounds: (0, 1000) for reasonable integer range.

    Examples:
        # Default bounded integer variable
        n = IntegerVariable("n")

        # Custom bounded integer variable
        count = IntegerVariable("count", bounds=(1, 100))
    """

    def __init__(self, name: str, bounds: Tuple[int, int] = (0, 1000)):
        """
        Initialize integer variable.

        Args:
            name: Variable name (must be non-empty string)
            bounds: Variable bounds as (min_val, max_val) tuple

        NASA Assert: bounds are integers and min <= max
        """
        production_assert(isinstance(bounds[0], int), 
                         f"Integer variable lower bound must be integer, got {type(bounds[0])}")
        production_assert(isinstance(bounds[1], int), 
                         f"Integer variable upper bound must be integer, got {type(bounds[1])}")

        super().__init__(name, bounds)

    def validate_value(self, value: Any) -> bool:
        """
        Validate integer value.

        Args:
            value: Value to validate

        Returns:
            bool: True if value is integer and within bounds

        NASA Assert: value is integer and within bounds
        """
        if not isinstance(value, int):
            return False

        if self.bounds is not None:
            return self.bounds[0] <= value <= self.bounds[1]

        return True

    def get_type_name(self) -> str:
        """Return variable type identifier."""
        return "integer"