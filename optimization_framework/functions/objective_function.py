"""
Objective Function Implementation

Provides ObjectiveFunction wrapper class with comprehensive validation
and variable dictionary protocol enforcement.
"""

import math
import traceback
from typing import Dict, Any, Callable
from ..utils.assertions import production_assert


class FunctionFrameworkError(Exception):
    """Base exception for function framework errors."""
    pass


class FunctionEvaluationError(FunctionFrameworkError):
    """Exception for function evaluation failures."""
    pass


class ObjectiveFunction:
    """
    Wrapper for objective functions with variable dict protocol.

    Ensures all objective functions follow consistent interface and
    provides comprehensive validation of inputs and outputs.
    """

    def __init__(self, func: Callable[[Dict], float], name: str = "objective"):
        """
        Initialize objective function wrapper.

        Args:
            func: Function that accepts variable_dict and returns float
            name: Function name for identification

        NASA Assert: func is callable, name is non-empty string
        """
        production_assert(callable(func), f"Objective function must be callable, got {type(func)}")
        production_assert(isinstance(name, str), f"Function name must be string, got {type(name)}")
        production_assert(len(name.strip()) > 0, "Function name cannot be empty")

        self.func = func
        self.name = name.strip()

    def evaluate(self, variable_dict: Dict) -> float:
        """
        Evaluate objective function with full validation.

        Args:
            variable_dict: Dictionary following variable dict protocol

        Returns:
            float: Objective function value

        NASA Assert: variable_dict follows protocol, return value is numeric
        """
        production_assert(self.validate_variable_dict(variable_dict), 
                         "Invalid variable dictionary format")

        try:
            result = self.func(variable_dict)
        except Exception as e:
            raise FunctionEvaluationError(f"Objective function evaluation failed: {str(e)}")

        production_assert(isinstance(result, (int, float)), 
                         f"Objective function must return numeric value, got {type(result)}")
        production_assert(not (isinstance(result, float) and math.isnan(result)), 
                         "Objective function returned NaN")

        return float(result)

    def validate_variable_dict(self, variable_dict: Dict) -> bool:
        """
        Validate variable dictionary format and contents.

        Args:
            variable_dict: Dictionary to validate

        Returns:
            bool: True if dictionary follows protocol

        NASA Assert: dict structure matches protocol, all variable objects validate their values
        """
        production_assert(isinstance(variable_dict, dict), 
                         f"Expected dict, got {type(variable_dict)}")
        production_assert(len(variable_dict) > 0, "Variable dictionary cannot be empty")

        for var_name, var_data in variable_dict.items():
            production_assert(isinstance(var_name, str), 
                             f"Variable name must be string, got {type(var_name)}")
            production_assert(isinstance(var_data, dict), 
                             f"Variable data must be dict, got {type(var_data)}")

            required_keys = ["value", "type", "bounds", "variable_object"]
            for key in required_keys:
                production_assert(key in var_data, 
                                 f"Missing required key '{key}' in variable '{var_name}'")

            # Validate variable object if available
            if "variable_object" in var_data and var_data["variable_object"] is not None:
                var_obj = var_data["variable_object"]
                if hasattr(var_obj, 'validate_value'):
                    production_assert(var_obj.validate_value(var_data["value"]), 
                                     f"Invalid value for variable '{var_name}': {var_data['value']}")

        return True