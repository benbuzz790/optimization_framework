"""
Base Variable class for optimization framework.

This module provides the abstract base Variable class that defines the interface
and common functionality for all optimization variables. All variable types
must implement the variable dictionary protocol and NASA-style validation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
from ..utils.assertions import production_assert


class Variable(ABC):
    """
    Abstract base class for optimization variables with NASA-style validation.

    All variables must implement type-specific validation and provide
    consistent interface for the variable dictionary protocol.

    The variable dictionary protocol format:
    {
        "var_name": {
            "value": <current_value>,
            "type": "continuous" | "integer" | "binary",
            "bounds": (min_val, max_val),
            "variable_object": <Variable instance>
        }
    }

    Attributes:
        name (str): Variable name (must be unique within a problem)
        bounds (Tuple[float, float]): Variable bounds (min_val, max_val)
    """

    def __init__(self, name: str, bounds: Optional[Tuple[float, float]] = None):
        """
        Initialize variable with production asserts.

        Args:
            name: Variable name (must be non-empty string)
            bounds: Optional bounds tuple (min_val, max_val)

        NASA Assert: name must be non-empty string, bounds must be valid tuple if provided
        """
        production_assert(isinstance(name, str), f"Variable name must be string, got {type(name)}")
        production_assert(len(name.strip()) > 0, "Variable name cannot be empty")

        if bounds is not None:
            production_assert(isinstance(bounds, tuple), f"Bounds must be tuple, got {type(bounds)}")
            production_assert(len(bounds) == 2, f"Bounds must have 2 elements, got {len(bounds)}")
            production_assert(bounds[0] <= bounds[1], f"Invalid bounds: {bounds[0]} > {bounds[1]}")

        self.name = name.strip()
        self.bounds = bounds

    @abstractmethod
    def validate_value(self, value: Any) -> bool:
        """
        Validate if value is acceptable for this variable type.

        Args:
            value: Value to validate

        Returns:
            bool: True if value is valid for this variable type

        NASA Assert: Subclasses must implement type-specific validation
        """
        pass

    def clip_to_bounds(self, value: Any) -> Any:
        """
        Clip value to variable bounds.

        Args:
            value: Value to clip

        Returns:
            Clipped value within bounds

        NASA Assert: bounds must exist before clipping
        """
        production_assert(self.bounds is not None, f"Cannot clip {self.name}: no bounds defined")

        # For clipping, we don't require the value to be valid beforehand
        # We just need it to be the right type for this variable
        if not isinstance(value, (int, float)):
            production_assert(False, f"Cannot clip non-numeric value for {self.name}: {value}")

        clipped = max(self.bounds[0], min(self.bounds[1], value))

        # For integer variables, ensure the result is an integer
        if self.get_type_name() == "integer":
            clipped = int(round(clipped))

        return clipped

    def to_dict_entry(self, value: Any) -> Dict:
        """
        Convert to variable dictionary entry format.

        Args:
            value: Current variable value

        Returns:
            Dict: Variable dictionary entry following protocol

        NASA Assert: value must be valid before conversion
        """
        production_assert(self.validate_value(value), 
                         f"Invalid value for {self.name}: {value}")

        return {
            "value": value,
            "type": self.get_type_name(),
            "bounds": self.bounds,
            "variable_object": self
        }

    @abstractmethod
    def get_type_name(self) -> str:
        """Return variable type identifier."""
        pass