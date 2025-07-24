"""
Constraint Function Implementation

Provides ConstraintFunction wrapper class with constraint satisfaction checking
and violation amount calculation.
"""

import math
from typing import Dict, Any, Callable
from ..utils.assertions import production_assert
from .objective_function import ObjectiveFunction, FunctionEvaluationError


class ConstraintViolationError(Exception):
    """Exception for constraint violation calculation errors."""
    pass


class ConstraintFunction:
    """
    Wrapper for constraint functions with variable dict protocol.

    Handles constraint evaluation, satisfaction checking, and violation calculation.
    Supports <=, >=, and == constraint types.
    """

    def __init__(self, func: Callable[[Dict], float], constraint_type: str, 
                 bound: float, name: str = "constraint"):
        """
        Initialize constraint function wrapper.

        Args:
            func: Function that accepts variable_dict and returns float
            constraint_type: One of "<=", ">=", "=="
            bound: Constraint bound value
            name: Constraint name for identification

        NASA Assert: func is callable, constraint_type is valid, bound is numeric
        """
        production_assert(callable(func), f"Constraint function must be callable, got {type(func)}")
        production_assert(constraint_type in ["<=", ">=", "=="], 
                         f"Invalid constraint type: {constraint_type}")
        production_assert(isinstance(bound, (int, float)), 
                         f"Constraint bound must be numeric, got {type(bound)}")
        production_assert(isinstance(name, str), f"Constraint name must be string, got {type(name)}")

        self.func = func
        self.constraint_type = constraint_type
        self.bound = float(bound)
        self.name = name.strip()

    def evaluate(self, variable_dict: Dict) -> float:
        """
        Evaluate constraint function.

        Args:
            variable_dict: Dictionary following variable dict protocol

        Returns:
            float: Constraint function value

        NASA Assert: variable_dict validation
        """
        # Reuse ObjectiveFunction validation logic
        obj_func = ObjectiveFunction(self.func, f"constraint_{self.name}")
        return obj_func.evaluate(variable_dict)

    def is_satisfied(self, variable_dict: Dict) -> bool:
        """
        Check if constraint is satisfied.

        Args:
            variable_dict: Dictionary following variable dict protocol

        Returns:
            bool: True if constraint is satisfied

        NASA Assert: evaluation successful before checking
        """
        try:
            value = self.evaluate(variable_dict)
        except Exception as e:
            raise ConstraintViolationError(f"Cannot check constraint satisfaction: {str(e)}")

        if self.constraint_type == "<=":
            return value <= self.bound
        elif self.constraint_type == ">=":
            return value >= self.bound
        else:  # "=="
            return abs(value - self.bound) < 1e-10

    def violation_amount(self, variable_dict: Dict) -> float:
        """
        Calculate constraint violation amount (0 if satisfied).

        Args:
            variable_dict: Dictionary following variable dict protocol

        Returns:
            float: Violation amount (0 if constraint satisfied)
        """
        try:
            value = self.evaluate(variable_dict)
        except Exception as e:
            raise ConstraintViolationError(f"Cannot calculate constraint violation: {str(e)}")

        if self.constraint_type == "<=":
            return max(0, value - self.bound)
        elif self.constraint_type == ">=":
            return max(0, self.bound - value)
        else:  # "=="
            return abs(value - self.bound)