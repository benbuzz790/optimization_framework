"""
Continuous Variable class for optimization framework.

This module provides the ContinuousVariable class for real-valued optimization
variables with full NASA-style validation and bounds handling.
"""

from typing import Tuple, Any
from .variable import Variable
from ..utils.assertions import production_assert


class ContinuousVariable(Variable):
    """
    Continuous real-valued variable.

    Accepts any numeric value (int or float) within specified bounds.
    Default bounds: (-inf, +inf) for unbounded optimization.

    Examples:
        # Unbounded continuous variable
        x = ContinuousVariable("x")

        # Bounded continuous variable
        y = ContinuousVariable("y", bounds=(-10.0, 10.0))
    """

    def __init__(self, name: str, bounds: Tuple[float, float] = (-float('inf'), float('inf'))):
        """
        Initialize continuous variable.

        Args:
            name: Variable name (must be non-empty string)
            bounds: Variable bounds as (min_val, max_val) tuple

        NASA Assert: bounds are numeric and min <= max
        """
        production_assert(isinstance(bounds[0], (int, float)), 
                         f"Lower bound must be numeric, got {type(bounds[0])}")
        production_assert(isinstance(bounds[1], (int, float)), 
                         f"Upper bound must be numeric, got {type(bounds[1])}")

        super().__init__(name, bounds)

    def validate_value(self, value: Any) -> bool:
        """
        Validate continuous value.

        Args:
            value: Value to validate

        Returns:
            bool: True if value is numeric and within bounds

        NASA Assert: value is numeric and within bounds
        """
        if not isinstance(value, (int, float)):
            return False

        if self.bounds is not None:
            return self.bounds[0] <= value <= self.bounds[1]

        return True

    def get_type_name(self) -> str:
        """Return variable type identifier."""
        return "continuous"